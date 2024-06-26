from utils import (
    np, zeros, array, arange, tobytes, append, np,
    modinv, M31, M31SQ, log2, point_add, to_extension_field,
    get_challenges, modinv_ext, merkelize_top_dimension,
    rbo_index_to_original, pad_to, m31_arith, ext_arith,
    eval_zpoly_at, projective_to_point, point_add_ext,
    fold, fold_ext, mul_ext
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

def mk_stark(get_next_state_vector,
             trace_width,
             constants,
             arguments,
             public_args,
             prebuilt_constants_tree=None,
             prefilled_trace=None):
    import time
    rounds, constants_width = constants.shape[:2]
    trace_length = 2**(rounds+1).bit_length()
    print('Trace length: {}'.format(trace_length))
    START = time.time()
    constants = pad_to(constants, trace_length)
    arguments = pad_to(arguments, trace_length)
    arguments_width = arguments.shape[1]

    if prefilled_trace is not None:
        trace = prefilled_trace
    else:
        trace = zeros((trace_length, trace_width))
        for i in range(rounds):
            trace[i+1] = get_next_state_vector(
                trace[i],
                constants[i],
                arguments[i],
                m31_arith
            )
    output_state = trace[rounds]
    print('Generated trace!', time.time() - START)
    trace_coeffs = fft(trace)
    trace_ext4 = inv_fft(pad_to(trace_coeffs, trace_length*4))
    constants_coeffs = fft(constants)
    constants_ext4 = inv_fft(pad_to(constants_coeffs, trace_length*4))
    arguments_coeffs = fft(arguments)
    arguments_ext4 = inv_fft(pad_to(arguments_coeffs, trace_length*4))

    # Trace must satisfy
    # C(T(x), T(x+G), K(x), A(x)) = Z(x) * H(x)
    rolled_trace4 = append(trace_ext4[4:], trace_ext4[:4])
    nsv = get_next_state_vector(
        trace_ext4.swapaxes(0, 1),
        constants_ext4.swapaxes(0, 1),
        arguments_ext4.swapaxes(0, 1),
        m31_arith
    ).swapaxes(0, 1)
    C_ext4 = (nsv - rolled_trace4) % M31
    ext4_domain = sub_domains[trace_length*4: trace_length*8]
    print('Generated size-4x polynomials', time.time() - START)
    # Extend to 8x
    #ext8_domain = sub_domains[trace_length*8: trace_length*16]
    #C_ext8 = inv_fft(pad_to(fft(C_ext4), trace_length*8))
    #assert confirm_max_degree(fft(C_ext8), trace_length*3)
    #trace_ext8 = inv_fft(pad_to(trace_coeffs, trace_length*8))
    #constants_ext8 = inv_fft(pad_to(constants_coeffs, trace_length*8))
    #arguments_ext8 = inv_fft(pad_to(arguments_coeffs, trace_length*8))
    #print('Generated some size-8x polynomials', time.time() - START)
    Va, Ia = public_args_to_vanish_and_interp(
        trace_length,
        public_args,
        arguments[array(public_args)],
        m31_arith
    )
    print('Generated Va,Ia', time.time() - START)
    Va = Va.reshape((Va.shape[0], 1))
    Ia_ext4 = inv_fft(pad_to(fft(Ia), trace_length*4))
    Va_ext4 = inv_fft(pad_to(fft(Va), trace_length*4))
    args_quotient_ext4 = (arguments_ext4 - Ia_ext4) * modinv(Va_ext4) % M31
    #assert confirm_max_degree(fft(args_quotient_ext8), trace_length)
    Vt, It = public_args_to_vanish_and_interp(
        trace_length,
        (0, rounds),
        trace[array((0, rounds))],
        m31_arith
    )
    Vt = Vt.reshape((Vt.shape[0], 1))
    It_ext4 = inv_fft(pad_to(fft(It), trace_length*4))
    Vt_ext4 = inv_fft(pad_to(fft(Vt), trace_length*4))
    trace_quotient_ext4 = (trace_ext4 - It_ext4) * modinv(Vt_ext4) % M31
    #assert confirm_max_degree(fft(trace_quotient_ext8), trace_length)
    #print('Generated size-8x polynomials', time.time() - START)
    Z_ext4 = eval_zpoly_at(
        trace_length,
        ext4_domain.reshape((trace_length*4, 1, 2)),
        m31_arith
    )
    H_ext4 = C_ext4 * modinv(Z_ext4) % M31
    #H_coeffs = fft(H_ext8)
    #assert confirm_max_degree(H_coeffs, trace_length*2+3)
    print('About to make trees!', time.time() - START)
    stack_ext4 = np.hstack((
        trace_quotient_ext4,
        args_quotient_ext4,
        constants_ext4,
        H_ext4
    ))
    TA_ext4 = stack_ext4[:,:trace_width + arguments_width]
    TA_tree = merkelize_top_dimension(TA_ext4)
    print('Generated first tree!', time.time() - START)
    if prebuilt_constants_tree is not None:
        K_tree = prebuilt_constants_tree
    else:
        K_tree = merkelize_top_dimension(constants_ext4)
    print('Generated second tree!', time.time() - START)
    # Fold and prove
    G = sub_domains[trace_length//2]
    bump = to_extension_field(sub_domains[trace_length*4])
    w = projective_to_point(get_challenges(TA_tree[1], M31, 4))
    w_plus_G = point_add_ext(w, to_extension_field(G))
    #print('T_at_w', bary_eval(to_extension_field(trace_ext8), w, ext_arith)[:3])
    #print('K_at_w', bary_eval(to_extension_field(constants_ext8), w, ext_arith)[:3])
    #print('A_at_w', bary_eval(to_extension_field(arguments_ext8), w, ext_arith)[:3])
    #print('T_at_w_plus_G', bary_eval(to_extension_field(trace_ext8), w_plus_G, ext_arith)[:3])
    #print('C_at_w', bary_eval(to_extension_field(C_ext8), w, ext_arith))[:3]
    #print('Z_at_w', bary_eval(to_extension_field(Z_ext8), w, ext_arith))
    #print('H_at_w', bary_eval(to_extension_field(H_ext8), w, ext_arith))[:3]
    w_bump = point_add_ext(w, bump)
    comb = np.hstack((
        stack_ext4[::2],
        append(stack_ext4[4::2],stack_ext4[:4:2]),
    ))
    S_bary = bary_eval(to_extension_field(comb), w_bump, ext_arith)
    stack_width = trace_width * 2 + constants_width + arguments_width
    S_at_w = S_bary[:stack_width]
    S_at_w_plus_G = S_bary[stack_width:]
    print("Computed evals at w and w+G", time.time() - START)
    entropy = TA_tree[1] + K_tree[1] + tobytes(S_at_w) + tobytes(S_at_w_plus_G)
    fold_factors = (
        get_challenges(entropy, M31, stack_width * 4)
        .reshape((stack_width, 4))
    )
    merged_poly = (
        fold(stack_ext4, fold_factors)
    ) % M31
    #merged_poly_coeffs = fft(merged_poly)
    #assert confirm_max_degree(merged_poly_coeffs, trace_length * 3 + 3)
    print('Generated merged poly!', time.time() - START)
    L3 = line_function(w, w_plus_G, ext4_domain, ext_arith)
    I3 = interpolant(
        w,
        fold_ext(S_at_w, fold_factors),
        w_plus_G,
        fold_ext(S_at_w_plus_G, fold_factors),
        ext4_domain,
        ext_arith
    )
    master_quotient = mul_ext(
        (merged_poly - I3) % M31, modinv_ext(L3)
    )
    print('Generated master_quotient!', time.time() - START)
    #master_quotient_coeffs = fft(master_quotient)
    #assert confirm_max_degree(master_quotient_coeffs, trace_length * 3)

    fri_proof = prove_low_degree(master_quotient)
    entropy = (
        b''.join(fri_proof["roots"]) +
        tobytes(fri_proof["final_values"])
    )
    challenges_raw = get_challenges(
        entropy, trace_length*4, NUM_CHALLENGES
    )
    fri_top_leaf_count = trace_length*4 >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw >> log2(fri_top_leaf_count)
    challenges = rbo_index_to_original(
        trace_length*4,
        challenges_top * FOLD_SIZE_RATIO + challenges_bottom
    )
    challenges_next = (challenges+4) % (trace_length*4)

    return {
        "output_state": output_state,
        "fri": fri_proof,
        "TA_root": TA_tree[1],
        "TA_branches": [get_branch(TA_tree, c) for c in challenges],
        "TA_leaves": TA_ext4[challenges],
        "TA_next_branches": [get_branch(TA_tree, c) for c in challenges_next],
        "TA_next_leaves": TA_ext4[challenges_next],
        "K_root": K_tree[1],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext4[challenges],
        "S_at_w": S_at_w,
        "S_at_w_plus_G": S_at_w_plus_G,
    }

def build_constants_tree(constants):
    rounds = constants.shape[0]
    trace_length = 2**rounds.bit_length()
    constants_coeffs = fft(pad_to(constants, trace_length))
    return merkelize_top_dimension(
        inv_fft(pad_to(constants_coeffs, trace_length*4))
    )

def get_vk(trace_width, constants, arguments_width, public_args_positions):
    rounds = constants.shape[0]
    return {
        "root": build_constants_tree(constants)[1],
        "rounds": rounds,
        "trace_width": trace_width,
        "constants_width": constants.shape[1],
        "arguments_width": arguments_width,
        "public_args_positions": public_args_positions,
    }

def verify_stark(get_next_state_vector, vk, public_args, proof):
    fri_proof = proof["fri"]
    assert verify_low_degree(fri_proof)
    len_evaluations = (
        fri_proof["final_values"].shape[0]
        << (FOLDS_PER_ROUND * len(fri_proof["roots"]))
    )
    TA_root = proof["TA_root"]
    TA_branches = proof["TA_branches"]
    TA_leaves = proof["TA_leaves"]
    TA_next_branches = proof["TA_next_branches"]
    TA_next_leaves = proof["TA_next_leaves"]
    S_at_w = proof["S_at_w"]
    S_at_w_plus_G = proof["S_at_w_plus_G"]
    K_root = vk["root"]
    K_branches = proof["K_branches"]
    K_leaves = proof["K_leaves"]
    rounds = vk["rounds"]
    constants_width = vk["constants_width"]
    arguments_width = vk["arguments_width"]
    public_args_positions = vk["public_args_positions"]
    trace_width = vk["trace_width"]
    trace_length = 2**(rounds+1).bit_length()
    G = sub_domains[trace_length//2]
    output_state = proof["output_state"]
    w = projective_to_point(get_challenges(TA_root, M31, 4))
    w_plus_G = point_add_ext(w, to_extension_field(G))
    Va, Ia = public_args_to_vanish_and_interp(
        trace_length,
        public_args_positions,
        public_args,
        m31_arith
    )
    Vt, It = public_args_to_vanish_and_interp(
        trace_length,
        (0, rounds),
        np.stack((zeros(trace_width), output_state)),
        m31_arith
    )
    T_at_w = (
        mul_ext(
            S_at_w[:trace_width],
            bary_eval(to_extension_field(Vt), w, ext_arith)
        )
        + bary_eval(to_extension_field(It), w, ext_arith)
    ) % M31
    T_at_w_plus_G = (
        mul_ext(
            S_at_w_plus_G[:trace_width],
            bary_eval(to_extension_field(Vt), w_plus_G, ext_arith)
        )
        + bary_eval(to_extension_field(It), w_plus_G, ext_arith)
    ) % M31
    A_at_w = (
        mul_ext(
            S_at_w[trace_width: trace_width + arguments_width],
            bary_eval(to_extension_field(Va), w, ext_arith)
        )
        + bary_eval(to_extension_field(Ia), w, ext_arith)
    ) % M31
    K_at_w = S_at_w[trace_width + arguments_width: -trace_width]
    H_at_w = S_at_w[-trace_width:]
    C_at_w = (
        get_next_state_vector(T_at_w, K_at_w, A_at_w, ext_arith)
        + M31 - T_at_w_plus_G
    ) % M31
    #print('T_at_w', T_at_w[:3])
    #print('K_at_w', K_at_w[:3])
    #print('A_at_w', A_at_w[:3])
    #print('T_at_w_plus_G', T_at_w_plus_G[:3])
    #print('C_at_w', C_at_w[:3])
    # Z = ext_domain[:,0,np.newaxis]
    # for i in range(1, log2(trace_length)):
    #     Z = (2 * Z**2 + M31 - 1) % M31
    Z_at_w = eval_zpoly_at(trace_length, w, ext_arith)
    computed_H_at_w = mul_ext(
        np.array(C_at_w),
        modinv_ext(Z_at_w)
    )
    assert np.array_equal(H_at_w, computed_H_at_w)
    entropy = b''.join(
        fri_proof["roots"] +
        [tobytes(x) for x in fri_proof["final_values"]]
    )
    challenges_raw = get_challenges(
        entropy, len_evaluations, NUM_CHALLENGES
    )
    fri_top_leaf_count = len_evaluations >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw // fri_top_leaf_count
    challenges = rbo_index_to_original(
        len_evaluations,
        challenges_top * FOLD_SIZE_RATIO + challenges_bottom
    )
    challenges_next = (challenges+4) % (trace_length*4)
    entropy = TA_root + K_root + tobytes(S_at_w) + tobytes(S_at_w_plus_G)
    stack_width = trace_width * 2 + constants_width + arguments_width
    fold_factors = (
        get_challenges(entropy, M31, stack_width * 4)
        .reshape((stack_width, 4))
    )
    Va_leaves, Ia_leaves = public_args_to_vanish_and_interp(
        trace_length,
        public_args_positions,
        public_args,
        m31_arith,
        out_domain=sub_domains[trace_length * 4 + challenges]
    )
    Vt_leaves, It_leaves = public_args_to_vanish_and_interp(
        trace_length,
        (0, rounds),
        np.stack((zeros(trace_width), output_state)),
        m31_arith,
        out_domain=append(
            sub_domains[trace_length * 4 + challenges],
            sub_domains[trace_length * 4 + challenges_next],
        )
    )
    Z_leaves = eval_zpoly_at(
        trace_length,
        sub_domains[trace_length * 4 + challenges],
        m31_arith
    )
    T_leaves = (
        TA_leaves[:,:trace_width]
        * Vt_leaves[:NUM_CHALLENGES].reshape((NUM_CHALLENGES, 1))
        + It_leaves[:NUM_CHALLENGES]
    ) % M31
    T_next_leaves = (
        TA_next_leaves[:,:trace_width]
        * Vt_leaves[NUM_CHALLENGES:].reshape((NUM_CHALLENGES, 1))
        + It_leaves[NUM_CHALLENGES:]
    ) % M31
    A_leaves = (
        TA_leaves[:,trace_width:]
        * Va_leaves.reshape((NUM_CHALLENGES, 1))
        + Ia_leaves[:NUM_CHALLENGES]
    ) % M31
    C_leaves = (get_next_state_vector(
        T_leaves.swapaxes(0, T_leaves.ndim-1),
        K_leaves.swapaxes(0, K_leaves.ndim-1),
        A_leaves.swapaxes(0, A_leaves.ndim-1),
        m31_arith
    ).swapaxes(0, T_leaves.ndim-1) + M31 - T_next_leaves) % M31
    H_leaves = (
        C_leaves * modinv(Z_leaves.reshape(Z_leaves.shape+(1,)))
    ) % M31
    inv_L3_leaves = modinv_ext(line_function(
        w,
        w_plus_G,
        sub_domains[trace_length * 4 + challenges],
        ext_arith
    ))
    M_at_w = (
        fold_ext(S_at_w, fold_factors)
    ) % M31
    M_at_w_plus_G = (
        fold_ext(S_at_w_plus_G, fold_factors)
    ) % M31
    I3_leaves = interpolant(
        w,
        M_at_w,
        w_plus_G,
        M_at_w_plus_G,
        sub_domains[trace_length * 4 + challenges],
        ext_arith
    )

    merged_leaves = (
        fold(np.hstack((TA_leaves, K_leaves, H_leaves)), fold_factors)
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
            TA_root, ci, tobytes(TA_leaves[i]), TA_branches[i])
        assert verify_branch(
            TA_root, nci, tobytes(TA_next_leaves[i]), TA_next_branches[i])
        assert verify_branch(
            K_root, ci, tobytes(K_leaves[i]), K_branches[i])
    return True
