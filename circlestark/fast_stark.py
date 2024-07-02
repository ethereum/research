from utils import (
    np, zeros, array, arange, tobytes, append, np,
    modinv, M31, M31SQ, log2, point_add, to_extension_field,
    get_challenges, modinv_ext, merkelize_top_dimension,
    rbo_index_to_original, pad_to, m31_arith, ext_arith,
    eval_zpoly_at, projective_to_point, point_add_ext,
    fold, fold_ext, mul_ext, confirm_max_degree, eval_monomial_at
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
    obj[-1] = (obj[-1] - coeffs[-1] * modinv(tweak_value)) % M31
    return obj

# Generate a STARK proof for a given claim
#
# get_next_state_vector  : Function that takes as as input trace column N and
#                          outputs trace column N+1. Must be degree <= 3
#
# constants              : Constants that get_next_state_vector has access to.
#                          This typically is two types: "opcodes" (constants
#                          reflecting which "part" of get_next_state_vector
#                          should be called, eg. t' = c1*f1(t) + c2*f2(t)...
#                          and other constants, eg. round constants for hashing
#
# arguments              : User-provided arguments to the function call
#
# public_args            : The arguments at which coordinates are public?
#
# prebuilt_constants_tree: The constants only need to be put into a Merkle tree
#                          once, so if you already did it, you can reuse the
#                          tree.
#
# prefilled_trace        : Often, there's a more efficient way to fill the
#                          trace than to call get_next_state_vector directly
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
    ext_degree = H_degree * 2
    F1, F2 = sub_domains[trace_length*2-2], sub_domains[trace_length*2-1]

    # Trace must satisfy
    # C(T(x), T(x+G), K(x), A(x)) = Z(x) * H(x)
    # Note that we multiply L[F1, F2] into C, to ensure that C does not
    # have to satisfy at the last coordinate in the set

    trace = tweak_last_row(trace)
    ext_domain = sub_domains[trace_length*ext_degree: trace_length*ext_degree*2]
    trace_ext = inv_fft(pad_to(fft(trace), trace_length*ext_degree))
    # We decompose both T and A as T=Qt*Vt+It and A=Qa*Va+Ia, which
    # allows the evaluations of each at a few specific points to be made
    # public
    V, I = public_args_to_vanish_and_interp(
        trace_length,
        public_args,
        trace[array(public_args)],
        m31_arith
    )
    V_ext = inv_fft(pad_to(fft(V), trace_length*ext_degree))
    I_ext = inv_fft(pad_to(fft(I), trace_length*ext_degree))
    #V1 = inv_fft(pad_to(fft(V), trace_length))
    #I1 = inv_fft(pad_to(fft(I), trace_length))
    print('Generated V,I', time.time() - START)
    trace_quotient_ext = (trace_ext - I_ext) * modinv(V_ext).reshape(V_ext.shape+(1,)) % M31
    constants_ext = inv_fft(pad_to(fft(constants), trace_length*ext_degree))
    rolled_trace = append(trace_ext[ext_degree:], trace_ext[:ext_degree])
    L_ext = line_function(F1, F2, ext_domain, m31_arith).reshape((trace_length*ext_degree, 1))
    np.cuda.synchronize()

    def f1():
        return merkelize_top_dimension(trace_quotient_ext)

    def f2():
        return check_constraint(
            trace_ext.swapaxes(0, 1),
            rolled_trace.swapaxes(0, 1),
            constants_ext.swapaxes(0, 1),
            m31_arith
        ).swapaxes(0, 1) * L_ext % M31

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the tasks to the executor
        future1 = executor.submit(f1)
        future2 = executor.submit(f2)

        # Get the results
        TQ_tree = future1.result()
        C_ext = future2.result()

    print('Generated tree and C_ext!', time.time() - START)

    output_width = C_ext.shape[1]
    #assert confirm_max_degree(fft(C_ext), (trace_length*(H_degree+1)+3))
    # C has degree < 3n, so n or 2n evaluations are not sufficient to
    # represent it; you need 4n (or 3n, but it's easier to work with powers
    # of two)

    # Because C is equal to 0 across the entire set of coordinates on which
    # the original trace was defined, except F1 (the wraparound point),
    # C * L[F2, F1] must be a multiple of Z, the simplest polynomial which
    # has that property. We compute H = (C * L) / Z, and then also quotient
    # it at {F2, F1}, to push it cleanly under degree 2n
    Z_ext = eval_zpoly_at(
        trace_length,
        ext_domain.reshape((trace_length*ext_degree, 1, 2)),
        m31_arith
    )
    # Note that H must have degree < 2n

    H_ext = C_ext * modinv(Z_ext) % M31
    H2_ext = H_ext
    #H_at_F1 = bary_eval(H_ext, F1, m31_arith)
    #H_at_F2 = bary_eval(H_ext, F2, m31_arith)
    #I2_ext = interpolant(F1, H_at_F1, F2, H_at_F2, ext_domain, m31_arith)
    #H2_ext = ((H_ext - I2_ext) * modinv(L_ext)) % M31
    #assert confirm_max_degree(fft(H2_ext), trace_length*H_degree)
    print('About to make trees!', time.time() - START)


    # Generate a tree of the constants if we have not yet
    if prebuilt_constants_tree is not None:
        K_tree = prebuilt_constants_tree
    else:
        K_tree = merkelize_top_dimension(constants_ext)
    print('Generated second tree!', time.time() - START)

    # Now, we generate a random point w, at which we evaluate everything in
    # stack_ext4
    G = sub_domains[trace_length//2]
    w = projective_to_point(get_challenges(TQ_tree[1], M31, 4))
    w_plus_G = point_add_ext(w, to_extension_field(G))
    #print('T_at_w', bary_eval(to_extension_field(trace_ext), w, ext_arith)[-3:])
    #print('K_at_w', bary_eval(to_extension_field(constants_ext), w, ext_arith)[-3:])
    #print('T_at_w_plus_G', bary_eval(to_extension_field(trace_ext), w_plus_G, ext_arith)[-3:])
    #print('L_at_w', bary_eval(to_extension_field(L_ext), w, ext_arith)[-3:])
    #print('C_at_w', bary_eval(to_extension_field(C_ext), w, ext_arith)[-3:])
    #print('Z_at_w', bary_eval(to_extension_field(Z_ext), w, ext_arith))
    #print('H_at_w', bary_eval(to_extension_field(H_ext), w, ext_arith)[-3:])
    #print('H2_at_w', bary_eval(to_extension_field(H2_ext), w, ext_arith)[-3:])

    # We put the polynomials we have together, commit to a Merkle tree of
    # {trace, arguments}
    stack_ext = np.hstack((
        trace_quotient_ext,
        constants_ext,
        H2_ext
    ))
    stack_width = trace_width + constants_width + output_width

    bump = to_extension_field(sub_domains[trace_length*ext_degree])
    w_bump = point_add_ext(w, bump)
    wpG_bump = point_add_ext(w_plus_G, bump)
    TQ_bary = mul_ext(
        (
            bary_eval(trace, w, ext_arith, True)
            - bary_eval(I, w, ext_arith, True)
        ) % M31,
        modinv_ext(bary_eval(V, w, ext_arith, True))
    )
    TQ_bary2 = mul_ext(
        (
            bary_eval(trace, w_plus_G, ext_arith, True)
            - bary_eval(I, w_plus_G, ext_arith, True)
        ) % M31,
        modinv_ext(bary_eval(V, w_plus_G, ext_arith, True))
    )
    K_bary = bary_eval(constants, w, ext_arith, True)
    K_bary2 = bary_eval(constants, w_plus_G, ext_arith, True)
    H2_ef = H2_ext[::2]
    H_bary = bary_eval(H2_ef, w_bump, ext_arith, True)
    H_bary2 = bary_eval(H2_ef, wpG_bump, ext_arith, True)
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
    merged_poly = (
        fold(stack_ext, fold_factors)
    ) % M31
    #merged_poly_coeffs = fft(merged_poly)
    #assert confirm_max_degree(merged_poly_coeffs, trace_length * H_degree)
    print('Generated merged poly!', time.time() - START)
    # Do the quotient trick, to prove that the evaluation we gave is
    # correct. Namely, prove: (random_linear_combination(S) - I) / L is a 
    # polynomial, where L is 0 at w w+G, and I is S_at_w and S_at_w_plus_G
    L3 = line_function(w, w_plus_G, ext_domain, ext_arith)
    I3 = interpolant(
        w,
        fold_ext(S_at_w, fold_factors),
        w_plus_G,
        fold_ext(S_at_w_plus_G, fold_factors),
        ext_domain,
        ext_arith
    )
    #assert np.array_equal(fold_ext(S_at_w, fold_factors), bary_eval(merged_poly, w, ext_arith))
    print(I3.shape, merged_poly.shape)
    master_quotient = mul_ext(
        (merged_poly - I3) % M31, modinv_ext(L3)
    )
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
        #"H_at_F1": H_at_F1,
        #"H_at_F2": H_at_F2,
        "K_root": K_tree[1],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext[challenges],
        "S_at_w": S_at_w,
        "S_at_w_plus_G": S_at_w_plus_G,
    }

def build_constants_tree(constants, H_degree=2):
    trace_length = constants.shape[0]
    constants_coeffs = fft(pad_to(constants, trace_length))
    return merkelize_top_dimension(
        inv_fft(pad_to(constants_coeffs, trace_length*H_degree*2))
    )

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
    #H_at_F1 = proof["H_at_F1"]
    #H_at_F2 = proof["H_at_F2"]
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
    w_plus_G = point_add_ext(w, to_extension_field(G))
    # We are going to compute H(w) = C(T(w), T(w+G), K(w), A(w)) / Z(w) and
    # check it (thereby checking "consistency" of S(w)), and then we also
    # do a similar check on the leaves
    V, I = public_args_to_vanish_and_interp(
        trace_length,
        public_row_positions,
        public_rows,
        m31_arith
    )

    T_at_w = (
        mul_ext(
            S_at_w[:trace_width],
            bary_eval(to_extension_field(V), w, ext_arith)
        )
        + bary_eval(to_extension_field(I), w, ext_arith)
    ) % M31
    T_at_w_plus_G = (
        mul_ext(
            S_at_w_plus_G[:trace_width],
            bary_eval(to_extension_field(V), w_plus_G, ext_arith)
        )
        + bary_eval(to_extension_field(I), w_plus_G, ext_arith)
    ) % M31
    K_at_w = S_at_w[trace_width: -output_width]
    H2_at_w = S_at_w[-output_width:]
    F1, F2 = sub_domains[trace_length*2-2], sub_domains[trace_length*2-1]
    L_at_w = line_function(F1, F2, w.reshape((1,2,4)), ext_arith)
    C_at_w = mul_ext(
        check_constraint(T_at_w, T_at_w_plus_G, K_at_w, ext_arith),
        L_at_w
    )

    Z_at_w = eval_zpoly_at(trace_length, w, ext_arith)
    H_at_w = mul_ext(
        C_at_w,
        modinv_ext(Z_at_w)
    )
    #I_at_w = interpolant(
    #    F1,
    #    to_extension_field(H_at_F1),
    #    F2,
    #    to_extension_field(H_at_F2),
    #    w.reshape((1,2,4)),
    #    ext_arith
    #)
    computed_H2_at_w = H_at_w
    #computed_H2_at_w = mul_ext((H_at_w - I_at_w) % M31, modinv_ext(L_at_w))[0]
    #print('TQ_at_w', S_at_w[:trace_width])
    #print('T_at_w', T_at_w[-3:])
    #print('K_at_w', K_at_w[-3:])
    #print('T_at_w_plus_G', T_at_w_plus_G[-3:])
    #print('L_at_w', L_at_w[-3:])
    #print('C_at_w', C_at_w[-3:])
    #print('Z_at_w', Z_at_w)
    #print('H_at_w', H_at_w[-3:])
    #print('H2_at_w', computed_H2_at_w[-3:], H2_at_w[-3:])
    assert np.array_equal(H2_at_w, computed_H2_at_w)
    # Now, the same check as above on the leaves
    stack_width = trace_width + constants_width + output_width
    fold_factors = (
        get_challenges(entropy, M31, stack_width * 4)
        .reshape((stack_width, 4))
    )
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

    V_leaves, I_leaves = public_args_to_vanish_and_interp(
        trace_length,
        public_row_positions,
        public_rows,
        m31_arith,
        out_domain=append(
            sub_domains[trace_length * ext_degree + challenges],
            sub_domains[trace_length * ext_degree + challenges_next],
        )
    )
    Z_leaves = eval_zpoly_at(
        trace_length,
        sub_domains[trace_length * ext_degree + challenges],
        m31_arith
    )
    L_leaves = line_function(
        F1,
        F2,
        sub_domains[trace_length * ext_degree + challenges],
        m31_arith
    ).reshape((challenges.shape[0], 1))
    T_leaves = (
        TQ_leaves[:,:trace_width]
        * V_leaves[:NUM_CHALLENGES].reshape((NUM_CHALLENGES, 1))
        + I_leaves[:NUM_CHALLENGES]
    ) % M31
    T_next_leaves = (
        TQ_next_leaves[:,:trace_width]
        * V_leaves[NUM_CHALLENGES:].reshape((NUM_CHALLENGES, 1))
        + I_leaves[NUM_CHALLENGES:]
    ) % M31
    C_leaves = (check_constraint(
        T_leaves.swapaxes(0, T_leaves.ndim-1),
        T_next_leaves.swapaxes(0, T_next_leaves.ndim-1),
        K_leaves.swapaxes(0, K_leaves.ndim-1),
        m31_arith
    ).swapaxes(0, T_leaves.ndim-1) * L_leaves) % M31
    H_leaves = (
        C_leaves * modinv(Z_leaves.reshape(Z_leaves.shape+(1,)))
    ) % M31
    #I_leaves = interpolant(
    #    F1,
    #    H_at_F1,
    #    F2,
    #    H_at_F2,
    #    sub_domains[trace_length * ext_degree + challenges],
    #    m31_arith
    #)
    H2_leaves = H_leaves
    #H2_leaves = (
    #    (H_leaves - I_leaves) * modinv(L_leaves)
    #) % M31
    # Except here we don't have an H to check against, instead we
    # keep going: here, we use the S_at_w and S_at_w_plus_G values
    # provided to quotient it, and then finally verify that the
    # quotient is a polynomial. The quotienting is not _strictly_
    # necessary, we could just FRI over H directly, but it adds
    # security (see: the DEEP-FRI papers)

    merged_leaves = (
        fold(np.hstack((TQ_leaves, K_leaves, H2_leaves)), fold_factors)
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
        ext_arith
    ))
    I3_leaves = interpolant(
        w,
        M_at_w,
        w_plus_G,
        M_at_w_plus_G,
        sub_domains[trace_length * ext_degree + challenges],
        ext_arith
    )
    computed_U_leaves = mul_ext(
        (merged_leaves + M31 - I3_leaves) % M31,
        inv_L3_leaves
    )
    U_leaves = fri_proof["leaf_values"][0][
        np.arange(NUM_CHALLENGES),
        challenges_bottom[np.arange(NUM_CHALLENGES)]
    ]
    assert np.array_equal(U_leaves, computed_U_leaves)

    for i in range(NUM_CHALLENGES):
        ci, nci = int(challenges[i]), int(challenges_next[i])
        assert verify_branch(
            TQ_root, ci, tobytes(TQ_leaves[i]), TQ_branches[i])
        assert verify_branch(
            TQ_root, nci, tobytes(TQ_next_leaves[i]), TQ_next_branches[i])
        assert verify_branch(
            K_root, ci, tobytes(K_leaves[i]), K_branches[i])
    return True
