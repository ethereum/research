from utils import (
    log2, to_extension_field,
    get_challenges, merkelize_top_dimension,
    rbo_index_to_original, pad_to,
    eval_zpoly_at, projective_to_point,
    fold, fold_ext, confirm_max_degree
)

from zorch.m31 import (
    zeros, array, arange, append, tobytes, add, sub, mul, cp as np,
    mul_ext, modinv, modinv_ext, sum as m31_sum, eq, iszero, M31
)

from precomputes import sub_domains

from fast_fft import fft, inv_fft, bary_eval

from fast_fri import (
    prove_low_degree, verify_low_degree,
    NUM_CHALLENGES, FOLDS_PER_ROUND, FOLD_SIZE_RATIO
)

from line_functions import (
    line_function, interpolant, public_args_to_vanish_and_interp
)

from merkle import merkelize, hash, get_branch, verify_branch

# Tweaks the last row of a trace or constraint object to reduce its degree
def tweak_last_row(obj):
    obj = np.copy(obj)
    coeffs = fft(obj)
    tweak_value = fft(append(zeros(obj.shape[0]-1), zeros(1) + 1))[-1]
    obj[-1] = sub(obj[-1], mul(coeffs[-1], modinv(tweak_value)))
    return obj

# Computes H(x) = C(T(x), T(next(x)), K(x)) / Z(x). This function gets
# used in multiple places, both the prover and the verifier
def compute_H(domain,
              T,
              T_offset,
              K,
              check_constraint,
              trace_length,
              is_extended,
              domain_is_multi=False):
    if is_extended:
        _mul = mul_ext
        _inv = modinv_ext
    else:
        _mul = mul
        _inv = modinv
    F1, F2 = sub_domains[trace_length*2-2], sub_domains[trace_length*2-1]
    Z = eval_zpoly_at(trace_length, domain, is_extended)
    if domain_is_multi:
        L = line_function(F1, F2, domain, is_extended)
        C = _mul(
            check_constraint(
                T.swapaxes(0,1),
                T_offset.swapaxes(0,1),
                K.swapaxes(0,1),
            is_extended).swapaxes(0,1),
            L.reshape(L.shape + (1,))
        )
        Z = Z.reshape(L.shape+(1,))
    else:
        L = line_function(F1, F2, domain.reshape((1,)+domain.shape), is_extended)
        C = _mul(
            check_constraint(T, T_offset, K, is_extended),
            L
        )

    # Uncomment for debugging
    # print('T_at_w', T[:3])
    # print('K_at_w', K[:3])
    # print('T_at_w_plus_G', T_offset[:3])
    # print('L_at_w', L[:3])
    # print('C_at_w', C[:3])
    # print('Z_at_w', Z)

    return _mul(C, _inv(Z))

# Generate a STARK proof for a given claim
#
# check_constraint(state, next_state, constraints, is_extended)
#
# Verifies that the constraint is satisfied at two adjacent rows of the trace.
# Must be degree <= H_degree+1
#
# trace: the computation trace
#
# constants: Constants that check_constraint has access to. This typically
#            includes opcodes and other constaints that differ row-by-row
#
# public_args: the rows of the trace that are revealed publicly
#
# prebuilt_constants_tree: The constants only need to be put into a Merkle tree
#                          once, so if you already did it, you can reuse the
#                          tree.
#
# H-degree: this plus one is the max degree of check_constraint. Must be a 
# power of two
def mk_stark(check_constraint,
             trace,
             constants,
             public_args,
             prebuilt_constants_tree=None,
             H_degree=2):
    import time
    rounds, constants_width = constants.shape[:2]
    trace_length = trace.shape[0]
    trace_width = trace.shape[1]
    print('Trace length: {}'.format(trace_length))
    START = time.time()
    constants = pad_to(constants, trace_length)

    # Trace must satisfy
    # C(T(x), T(x+G), K(x), A(x)) = Z(x) * H(x)
    # Note that we multiply L[F1, F2] into C, to ensure that C does not
    # have to satisfy at the last coordinate in the set

    # We use the last row of the trace to make its degree N-1 rather than N.
    # This keeps deg(H) later on comfortably less than N*H_degree-1, which
    # makes it more efficient to bary-evaluate
    trace = tweak_last_row(trace)
    # The larger domain on which we run our polynomial math
    ext_degree = H_degree * 2
    ext_domain = sub_domains[
        trace_length*ext_degree:
        trace_length*ext_degree*2
    ]
    trace_ext = inv_fft(pad_to(fft(trace), trace_length*ext_degree))
    print('Generated trace extension', time.time() - START)
    # Decompose the trace into the public part and the private part:
    # trace = public * V + private. We commit to the private part, and show
    # the public part in the clear
    V, I = public_args_to_vanish_and_interp(
        trace_length,
        public_args,
        trace[array(public_args)],
        False
    )
    V_ext = inv_fft(pad_to(fft(V), trace_length*ext_degree))
    I_ext = inv_fft(pad_to(fft(I), trace_length*ext_degree))
    print('Generated V,I', time.time() - START)
    trace_quotient_ext = mul(
        sub(trace_ext, I_ext),
        modinv(V_ext).reshape(V_ext.shape+(1,))
    )
    constants_ext = inv_fft(pad_to(fft(constants), trace_length*ext_degree))
    rolled_trace_ext = append(trace_ext[ext_degree:], trace_ext[:ext_degree])
    # Zero on the last two columns of the trace. We multiply this into C
    # to make it zero across the entire trace

    # We Merkelize the trace quotient (CPU-dominant) and compute C
    # (GPU-dominant) in parallel
    def f1():
        return merkelize_top_dimension(trace_quotient_ext)

    def f2():
        return compute_H(
            ext_domain,
            trace_ext,
            rolled_trace_ext,
            constants_ext,
            check_constraint,
            trace_length,
            is_extended=False,
            domain_is_multi=True
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the tasks to the executor
        future1 = executor.submit(f1)
        future2 = executor.submit(f2)
    
        # Get the results
        TQ_tree = future1.result()
        H_ext = future2.result()
    # TQ_tree = f1()
    # H_ext = f2()

    print('Generated tree and C_ext!', time.time() - START)

    #H_coeffs = fft(H_ext)
    #print(np.where(H_coeffs[trace_length * (H_degree + 1)] % M31 != 0), H_ext.shape)
    #assert confirm_max_degree(H_coeffs, trace_length * H_degree)

    output_width = H_ext.shape[1]
    if prebuilt_constants_tree is not None:
        K_tree = prebuilt_constants_tree
    else:
        K_tree = merkelize_top_dimension(constants_ext)

    # Now, we generate a random point w, at which we evaluate our polynomials
    G = sub_domains[trace_length//2]
    w = projective_to_point(get_challenges(TQ_tree[1], M31, 4))
    w_plus_G = w + G.to_extended()

    # Trace quotient, at w and w+G
    TQ_bary = mul_ext(
        sub(
            bary_eval(trace, w, True, True),
            bary_eval(I, w, True, True)
        ),
        modinv_ext(bary_eval(V, w, True, True))
    )
    TQ_bary2 = mul_ext(
        sub(
            bary_eval(trace, w_plus_G, True, True),
            bary_eval(I, w_plus_G, True, True)
        ),
        modinv_ext(bary_eval(V, w_plus_G, True, True))
    )
    # Constants, at w and w+G
    K_bary = bary_eval(constants, w, True, True)
    K_bary2 = bary_eval(constants, w_plus_G, True, True)

    # H, at w and w+G. We _could_ also compute it with compute_H, but
    # somehow a bary-evaluation is faster (!!) than calling the function
    bump = sub_domains[trace_length*ext_degree].to_extended()
    w_bump = w + bump
    wpG_bump = w_plus_G + bump
    H_ef = H_ext[::2]
    H_bary = bary_eval(H_ef, w_bump, True, True)
    H_bary2 = bary_eval(H_ef, wpG_bump, True, True)
    stack_ext = np.hstack((
        trace_quotient_ext,
        constants_ext,
        H_ext
    ))
    stack_width = trace_width + constants_width + output_width
    S_at_w = append(TQ_bary, K_bary, H_bary)
    S_at_w_plus_G = append(TQ_bary2, K_bary2, H_bary2)
    # Compute a random linear combination of everything in stack_ext4, using
    # S_at_w and S_at_w_plus_G as entropy
    print("Computed evals at w and w+G", time.time() - START)
    entropy = TQ_tree[1] + K_tree[1] + tobytes(S_at_w) + tobytes(S_at_w_plus_G)
    fold_factors = (
        get_challenges(entropy, M31, stack_width * 4)
        .reshape((stack_width, 4))
    )
    #assert eq(
    #    bary_eval(stack_ext, w, True, True),
    #    S_at_w
    #)
    merged_poly = (
        fold(stack_ext, fold_factors)
    ) % M31
    #merged_poly_coeffs = fft(merged_poly)
    #assert confirm_max_degree(merged_poly_coeffs, trace_length * H_degree)
    print('Generated merged poly!', time.time() - START)
    # Do the quotient trick, to prove that the evaluation we gave is
    # correct. Namely, prove: (random_linear_combination(S) - I) / L is a 
    # polynomial, where L is 0 at w w+G, and I is S_at_w and S_at_w_plus_G
    L3 = line_function(w, w_plus_G, ext_domain, True)
    I3 = interpolant(
        w,
        fold_ext(S_at_w, fold_factors),
        w_plus_G,
        fold_ext(S_at_w_plus_G, fold_factors),
        ext_domain,
        True
    )
    #assert eq(
    #    fold_ext(S_at_w, fold_factors),
    #    bary_eval(merged_poly, w, True)
    #)
    master_quotient = mul_ext(sub(merged_poly, I3), modinv_ext(L3))
    print('Generated master_quotient!', time.time() - START)
    #master_quotient_coeffs = fft(master_quotient)
    #assert confirm_max_degree(master_quotient_coeffs, trace_length * H_degree)

    # Generate a FRI proof of (random_linear_combination(S) - I) / L
    fri_proof = prove_low_degree(master_quotient, extra_entropy=entropy)
    fri_entropy = (
        entropy +
        b''.join(fri_proof["roots"]) +
        tobytes(fri_proof["final_values"])
    )
    challenges_raw = get_challenges(
        fri_entropy, trace_length*ext_degree, NUM_CHALLENGES
    )
    fri_top_leaf_count = trace_length*ext_degree >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw >> log2(fri_top_leaf_count)
    challenges = rbo_index_to_original(
        trace_length*ext_degree,
        challenges_top * FOLD_SIZE_RATIO + challenges_bottom
    )
    challenges_next = (challenges+ext_degree) % (trace_length*ext_degree)

    return {
        "fri": fri_proof,
        "TQ_root": TQ_tree[1],
        "TQ_branches": [get_branch(TQ_tree, c) for c in challenges],
        "TQ_leaves": trace_quotient_ext[challenges],
        "TQ_next_branches": [get_branch(TQ_tree, c) for c in challenges_next],
        "TQ_next_leaves": trace_quotient_ext[challenges_next],
        "K_root": K_tree[1],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext[challenges],
        "S_at_w": S_at_w,
        "S_at_w_plus_G": S_at_w_plus_G,
    }

# Generate the Merkle tree of constants
def build_constants_tree(constants, H_degree=2):
    trace_length = constants.shape[0]
    constants_coeffs = fft(pad_to(constants, trace_length))
    return merkelize_top_dimension(
        inv_fft(pad_to(constants_coeffs, trace_length*H_degree*2))
    )

# Generate the verification key (basically the Merkle root of the
# constants, plus some params)
def get_vk(trace_shape, constants, output_width, public_row_positions, H_degree=2):
    rounds = constants.shape[0]
    return {
        "root": build_constants_tree(constants, H_degree)[1],
        "trace_length": trace_shape[0],
        "trace_width": trace_shape[1],
        "constants_width": constants.shape[1],
        "output_width": output_width,
        "public_row_positions": public_row_positions,
        "ext_degree": H_degree*2
    }

def verify_stark(check_constraint, vk, public_rows, proof):
    fri_proof = proof["fri"]
    len_evaluations = (
        fri_proof["final_values"].shape[0]
        << (FOLDS_PER_ROUND * len(fri_proof["roots"]))
    )
    TQ_root = proof["TQ_root"]
    TQ_branches = proof["TQ_branches"]
    TQ_leaves = proof["TQ_leaves"]
    TQ_next_branches = proof["TQ_next_branches"]
    TQ_next_leaves = proof["TQ_next_leaves"]
    S_at_w = proof["S_at_w"]
    S_at_w_plus_G = proof["S_at_w_plus_G"]
    K_root = vk["root"]
    output_width = vk["output_width"]
    K_branches = proof["K_branches"]
    K_leaves = proof["K_leaves"]
    ext_degree = vk["ext_degree"]
    # Generate the extra entropy that was used to generate the FRI proof
    entropy = TQ_root + K_root + tobytes(S_at_w) + tobytes(S_at_w_plus_G)
    # Verify that proof
    assert verify_low_degree(fri_proof, extra_entropy=entropy)
    constants_width = vk["constants_width"]
    public_row_positions = vk["public_row_positions"]
    trace_width = vk["trace_width"]
    trace_length = vk["trace_length"]
    G = sub_domains[trace_length//2]
    # Pick the same two random points as the prover
    w = projective_to_point(get_challenges(TQ_root, M31, 4))
    w_plus_G = w + G.to_extended()
    # We are going to compute H(w) = C(T(w), T(w+G), K(w), A(w)) / Z(w) and
    # check it (thereby checking "consistency" of S(w)), and then we also
    # do a similar check on the leaves
    V, I = public_args_to_vanish_and_interp(
        trace_length,
        public_row_positions,
        public_rows,
        False
    )
    T_at_w = add(
        mul_ext(
            S_at_w[:trace_width],
            bary_eval(to_extension_field(V), w, True)
        ),
        bary_eval(to_extension_field(I), w, True)
    )
    T_at_w_plus_G = add(
        mul_ext(
            S_at_w_plus_G[:trace_width],
            bary_eval(to_extension_field(V), w_plus_G, True)
        ),
        bary_eval(to_extension_field(I), w_plus_G, True)
    )
    K_at_w = S_at_w[trace_width: -output_width]
    H_at_w = S_at_w[-output_width:]

    assert eq(
        H_at_w,
        compute_H(
            w,
            T_at_w,
            T_at_w_plus_G,
            K_at_w,
            check_constraint,
            trace_length,
            True
        )
    )
    # Now, the same check as above on the leaves
    fri_entropy = entropy + b''.join(
        fri_proof["roots"] +
        [tobytes(x) for x in fri_proof["final_values"]]
    )
    challenges_raw = get_challenges(
        fri_entropy, len_evaluations, NUM_CHALLENGES
    )
    fri_top_leaf_count = len_evaluations >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw // fri_top_leaf_count
    challenges = rbo_index_to_original(
        len_evaluations,
        challenges_top * FOLD_SIZE_RATIO + challenges_bottom
    )
    challenges_next = (challenges+ext_degree) % (trace_length*ext_degree)

    F1, F2 = sub_domains[trace_length*2-2], sub_domains[trace_length*2-1]
    V_leaves, I_leaves = public_args_to_vanish_and_interp(
        trace_length,
        public_row_positions,
        public_rows,
        False,
        out_domain=sub_domains[append(
            trace_length * ext_degree + challenges,
            trace_length * ext_degree + challenges_next,
        )]
    )
    T_leaves = add(
        mul(
            TQ_leaves[:,:trace_width],
            V_leaves[:NUM_CHALLENGES].reshape((NUM_CHALLENGES, 1))
        ),
        I_leaves[:NUM_CHALLENGES]
    )
    T_next_leaves = add(
        mul(
            TQ_next_leaves[:,:trace_width],
            V_leaves[NUM_CHALLENGES:].reshape((NUM_CHALLENGES, 1))
        ),
        I_leaves[NUM_CHALLENGES:]
    )

    H_leaves = compute_H(
        sub_domains[trace_length * ext_degree + challenges],
        T_leaves,
        T_next_leaves,
        K_leaves,
        check_constraint,
        trace_length,
        False,
        domain_is_multi=True
    )

    # Except here we don't have an H to check against, instead we
    # keep going: here, we use the S_at_w and S_at_w_plus_G values
    # provided to quotient it, and then finally verify that the
    # quotient is a polynomial. The quotienting is not _strictly_
    # necessary, we could just FRI over H directly, but it adds
    # security (see: the DEEP-FRI papers)
    stack_width = trace_width + constants_width + output_width
    fold_factors = (
        get_challenges(entropy, M31, stack_width * 4)
        .reshape((stack_width, 4))
    )

    merged_leaves = (
        fold(np.hstack((TQ_leaves, K_leaves, H_leaves)), fold_factors)
    )

    M_at_w = (
        fold_ext(S_at_w, fold_factors)
    ) % M31
    M_at_w_plus_G = (
        fold_ext(S_at_w_plus_G, fold_factors)
    ) % M31
    inv_L3_leaves = modinv_ext(line_function(
        w,
        w_plus_G,
        sub_domains[trace_length * ext_degree + challenges],
        True
    ))
    I3_leaves = interpolant(
        w,
        M_at_w,
        w_plus_G,
        M_at_w_plus_G,
        sub_domains[trace_length * ext_degree + challenges],
        True
    )
    computed_U_leaves = mul_ext(
        (merged_leaves + M31 - I3_leaves) % M31,
        inv_L3_leaves
    )
    U_leaves = fri_proof["leaf_values"][0][
        np.arange(NUM_CHALLENGES),
        challenges_bottom[np.arange(NUM_CHALLENGES)]
    ]
    assert eq(U_leaves, computed_U_leaves)

    for i in range(NUM_CHALLENGES):
        ci, nci = int(challenges[i]), int(challenges_next[i])
        assert verify_branch(
            TQ_root, ci, tobytes(TQ_leaves[i]), TQ_branches[i])
        assert verify_branch(
            TQ_root, nci, tobytes(TQ_next_leaves[i]), TQ_next_branches[i])
        assert verify_branch(
            K_root, ci, tobytes(K_leaves[i]), K_branches[i])
    return True
