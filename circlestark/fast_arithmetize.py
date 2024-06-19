from fast_fft import (
    np, modinv, M31, M31SQ, log2, fft, inv_fft, sub_domains,
    bary_eval, point_add, to_extension_field, bary_eval_ext,
    zeros, array, arange, tobytes, append
)
from fast_fri import (
    get_challenges, extension_field_mul, extension_point_add,
    modinv_ext, prove_low_degree, verify_low_degree,
    rbo_index_to_original, merkelize_top_dimension
)
from merkle import merkelize, hash, get_branch, verify_branch

NUM_CHALLENGES = 80
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND

def pad_to(arr, new_len):
    padding = zeros((new_len - arr.shape[0],) + arr.shape[1:])
    return append(arr, padding)

def confirm_max_degree(coeffs, bound):
    #print(coeffs, bound, coeffs[:bound+4])
    return np.array_equal(coeffs[bound:], np.zeros_like(coeffs[bound:]))

def line_function(P1, P2, domain):
    a = (P2[1] + M31 - P1[1]) % M31
    b = (P1[0] + M31 - P2[0]) % M31
    c = (M31SQ + P2[0] * P1[1] - P1[0] * P2[1]) % M31
    return (a * domain[:,0] + b * domain[:,1] + c) % M31

def line_function_ext(P1, P2, domain):
    a = (P2[1] + M31 - P1[1]) % M31
    b = (P1[0] + M31 - P2[0]) % M31
    c = (
        extension_field_mul(P2[0], P1[1])
        + M31 - extension_field_mul(P1[0], P2[1])
    ) % M31
    if len(domain.shape) == 2:
        x = domain[:,0].reshape((domain.shape[0], 1))
        y = domain[:,1].reshape((domain.shape[0], 1))
        return (a * x + b * y + c) % M31
    else:
        return (
            extension_field_mul(a, domain[:,0])
            + extension_field_mul(b, domain[:,1])
            + c
        ) % M31

def interpolant(P1, v1, P2, v2, domain):
    if P1[0] == P2[0]:
        y = domain[:,1].reshape((domain.shape[0], 1))
        slope = (v2 + M31 - v1) * modinv(P2[1] + M31 - P1[1]) % M31
        return v1 + (y + M31 - P1[1]) * slope % M31
    else:
        x = domain[:,0].reshape((domain.shape[0], 1))
        slope = (v2 + M31 - v1) * modinv(P2[0] + M31 - P1[0]) % M31
        return v1 + (x + M31 - P1[0]) * slope % M31

def interpolant_ext(P1, v1, P2, v2, domain):
    assert v1.shape[-1] == v2.shape[-1] == 4
    if np.array_equal(P1[0], P2[0]):
        coord1, coord2 = P1[1], P2[1]
        domain_coords = domain[:,1]
    else:
        coord1, coord2 = P1[0], P2[0]
        domain_coords = domain[:,0]

    slope = extension_field_mul(
        (v2 + M31 - v1) % M31,
        modinv_ext((coord2 + M31 - coord1) % M31)
    )
    if len(domain_coords.shape) == 1:
        domain_coords = to_extension_field(domain_coords)
    return (
        v1 + extension_field_mul(domain_coords + M31 - coord1, slope)
    ) % M31

def chunkify(values):
    o = values.astype(np.uint32).tobytes()
    width = values[0].size * 4
    return [o[i:i+width] for i in range(0, len(o), width)]

def projective_to_point(t):
    t2 = extension_field_mul(t, t)
    inv_1pt2 = modinv_ext((one + t2) % M31)
    return np.stack((
        extension_field_mul((one + M31 - t2) % M31, inv_1pt2),
        extension_field_mul((2 * t) % M31, inv_1pt2)
    ))

def mulM31(x, y):
    return x * y % M31

def fold(vector, fold_factors):
    return np.sum(
        vector.reshape(vector.shape+(1,)) * fold_factors % M31,
        axis=-2
    ) % M31

def fold_ext(vector, fold_factors):
    return np.sum(extension_field_mul(vector, fold_factors), axis=-2) % M31

one = array([1,0,0,0])
m31_arith = (
    1,
    lambda *x: sum(x) % M31,
    lambda x,y: x*y % M31
)
ext_arith = (
    one,
    lambda *x: sum(x) % M31,
    extension_field_mul
)

def mk_stark(get_next_state_vector, start_state, constants, zero_outside_trace=False):
    import time
    rounds, constants_width = constants.shape[:2]
    width = start_state.shape[0]
    trace_length = 2**(rounds+1).bit_length()
    print('Trace length: {}'.format(trace_length))
    trace = zeros((trace_length,) + start_state.shape)
    trace[0] = start_state
    START = time.time()
    bound = (rounds+1 if zero_outside_trace else trace_length-1)
    constants = pad_to(constants, trace_length)
    for i in range(bound):
        trace[i+1] = get_next_state_vector(
            trace[i],
            constants[i],
            m31_arith
        )
    output_state = trace[rounds]
    print('Generated trace!', time.time() - START)
    trace_coeffs = fft(trace)
    trace_ext4 = inv_fft(pad_to(trace_coeffs, trace_length*4))
    trace_ext = inv_fft(pad_to(trace_coeffs, trace_length*8))
    constants_coeffs = fft(constants)
    constants_ext4 = inv_fft(pad_to(constants_coeffs, trace_length*4))
    constants_ext = inv_fft(pad_to(constants_coeffs, trace_length*8))
    #assert confirm_max_degree(fft(constants_ext), trace_length)
    #assert confirm_max_degree(fft(trace_ext), trace_length)
    ext_domain = sub_domains[trace_length*8: trace_length*16]
    start_point = sub_domains[trace_length]
    output_point = sub_domains[trace_length + rounds]
    nm1_point = sub_domains[trace_length*2-1]
    nm2_point = sub_domains[trace_length*2-2]

    # Trace must satisfy
    #   C(T[x], T[x+G]) * (X - p0)
    # = Z(x) * H(x)
    rolled_trace4 = append(trace_ext4[4:], trace_ext4[:4])
    nsv = get_next_state_vector(
        trace_ext4.swapaxes(0, 1),
        constants_ext4.swapaxes(0, 1),
        m31_arith
    ).swapaxes(0, 1)
    C4 = (nsv + M31 - rolled_trace4) % M31
    C = inv_fft(pad_to(fft(C4), trace_length*8))
    print('Generated C!', time.time() - START)
    #assert confirm_max_degree(fft(np.roll(trace_ext, -8, axis=0)), trace_length)
    #assert confirm_max_degree(fft(C), trace_length*3+1)
    C_exempt_start = C * (
        #ext_domain[:,0,np.newaxis]
        #+ M31 - start_point[0]
        line_function(nm1_point, nm2_point, ext_domain)
        .reshape((trace_length*8, 1))
    ) % M31
    #C_coeffs = fft(C_exempt_start)
    #assert confirm_max_degree(C_coeffs, trace_length*3+3)
    Z = ext_domain[:,0].reshape((trace_length*8, 1))
    for i in range(1, log2(trace_length)):
        Z = (2 * Z**2 + M31 - 1) % M31
    H = C_exempt_start * modinv(Z) % M31
    #H_coeffs = fft(H)
    #assert confirm_max_degree(H_coeffs, trace_length*2+3)
    I = interpolant(
        start_point, start_state,
        output_point, output_state,
        ext_domain
    )
    L = line_function(start_point, output_point, ext_domain).reshape((trace_length*8, 1))
    T_quotient = ((trace_ext + M31 - I) * modinv(L)) % M31
    #assert confirm_max_degree(fft(T_quotient), trace_length)
    print('About to make trees!', time.time() - START)
    T_tree = merkelize_top_dimension(T_quotient)
    K_tree = merkelize_top_dimension(constants_ext)
    print('Generated trees!', time.time() - START)
    # Fold and prove
    G = sub_domains[trace_length//2]
    bump = to_extension_field(sub_domains[trace_length*8])
    w = projective_to_point(get_challenges(T_tree[1], M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
    w_bump = extension_point_add(w, bump)
    comb = np.hstack((
        H[::2],
        append(H[8::2],H[:8:2]),
        T_quotient[::2],
        append(T_quotient[8::2],T_quotient[:8:2]),
        constants_ext[::2],
        append(constants_ext[8::2],constants_ext[:8:2])
    ))
    comb_bary = bary_eval_ext(to_extension_field(comb), w_bump)
    H_at_w = comb_bary[:width]
    H_at_w_plus_G = comb_bary[width:width*2]
    TQ_at_w = comb_bary[width*2:width*3]
    TQ_at_w_plus_G = comb_bary[width*3:width*4]
    K_at_w = comb_bary[width*4:width*4+constants_width]
    K_at_w_plus_G = comb_bary[width*4+constants_width:]
    print("Computed evals at w and w+G", time.time() - START)
    entropy = T_tree[1] + b''.join(
        tobytes(H_at_w) for x in
        (TQ_at_w, TQ_at_w_plus_G, H_at_w, H_at_w_plus_G, K_at_w, K_at_w_plus_G)
    )
    fold_factors = (
        get_challenges(T_tree[1], M31, (width * 2 + constants_width) * 4)
        .reshape((width * 2 + constants_width, 4))
    )
    merged_poly = (
        fold(T_quotient, fold_factors[:width]) + 
        fold(H, fold_factors[width: width*2]) +
        fold(constants_ext, fold_factors[width*2:])
    ) % M31
    #merged_poly_coeffs = fft(merged_poly)
    #assert confirm_max_degree(merged_poly_coeffs, trace_length * 3 + 3)
    print('Generated merged poly!', time.time() - START)
    L3 = line_function_ext(w, w_plus_G, ext_domain)
    I3 = interpolant_ext(
        w,
        (
            fold_ext(TQ_at_w, fold_factors[:width])
            + fold_ext(H_at_w, fold_factors[width:width*2])
            + fold_ext(K_at_w, fold_factors[width*2:])
        ) % M31,
        w_plus_G,
        (
            fold_ext(TQ_at_w_plus_G, fold_factors[:width])
            + fold_ext(H_at_w_plus_G, fold_factors[width:width*2])
            + fold_ext(K_at_w_plus_G, fold_factors[width*2:])
        ) % M31,
        ext_domain
    )
    master_quotient = extension_field_mul(
        (merged_poly + M31 - I3) % M31,
        modinv_ext(L3)
    )
    print('Generated master_quotient!', time.time() - START)
    #master_quotient_coeffs = fft(master_quotient)
    #assert confirm_max_degree(master_quotient_coeffs, trace_length * 3 + 3)

    fri_proof = prove_low_degree(master_quotient)
    entropy = (
        b''.join(fri_proof["roots"]) +
        tobytes(fri_proof["final_values"])
    )
    challenges_raw = get_challenges(
        entropy, trace_length*8, NUM_CHALLENGES
    )
    fri_top_leaf_count = trace_length*8 >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw >> log2(fri_top_leaf_count)
    challenges = rbo_index_to_original(
        trace_length*8,
        challenges_top * 8 + challenges_bottom
    )
    challenges_next = (challenges+8) % (trace_length*8)

    return {
        "output_state": output_state,
        "fri": fri_proof,
        "TQ_root": T_tree[1],
        "TQ_branches": [get_branch(T_tree, c) for c in challenges],
        "TQ_leaves": T_quotient[challenges],
        "TQ_next_branches": [get_branch(T_tree, c) for c in challenges_next],
        "TQ_next_leaves": T_quotient[challenges_next],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext[challenges],
        "H_at_w": H_at_w,
        "H_at_w_plus_G": H_at_w_plus_G,
        "TQ_at_w": TQ_at_w,
        "TQ_at_w_plus_G": TQ_at_w_plus_G,
        "K_at_w": K_at_w,
        "K_at_w_plus_G": K_at_w_plus_G
    }

def get_vk(constants):
    rounds = constants.shape[0]
    trace_length = 2**rounds.bit_length()
    constants_coeffs = fft(pad_to(constants, trace_length))
    constants_ext = inv_fft(pad_to(constants_coeffs, trace_length*8))
    return {
        "root": merkelize_top_dimension(constants_ext)[1],
        "rounds": rounds,
        "constants_width": constants.shape[1]
    }

def verify_stark(get_next_state_vector, vk, start_state, proof):
    fri_proof = proof["fri"]
    assert verify_low_degree(fri_proof)
    len_evaluations = (
        fri_proof["final_values"].shape[0]
        << (FOLDS_PER_ROUND * len(fri_proof["roots"]))
    )
    TQ_branches = proof["TQ_branches"]
    TQ_leaves = proof["TQ_leaves"]
    TQ_next_branches = proof["TQ_next_branches"]
    TQ_next_leaves = proof["TQ_next_leaves"]
    K_branches = proof["K_branches"]
    K_leaves = proof["K_leaves"]
    TQ_at_w = proof["TQ_at_w"]
    TQ_at_w_plus_G = proof["TQ_at_w_plus_G"]
    H_at_w = proof["H_at_w"]
    H_at_w_plus_G = proof["H_at_w_plus_G"]
    K_at_w = proof["K_at_w"]
    K_at_w_plus_G = proof["K_at_w_plus_G"]
    K_root = vk["root"]
    rounds = vk["rounds"]
    constants_width = vk["constants_width"]
    width = start_state.shape[0]
    trace_length = 2**(rounds+1).bit_length()
    G = sub_domains[trace_length//2]
    start_point = sub_domains[trace_length]
    output_point = sub_domains[trace_length + rounds]
    nm1_point = sub_domains[trace_length*2-1]
    nm2_point = sub_domains[trace_length*2-2]
    output_state = proof["output_state"]
    TQ_root = proof["TQ_root"]
    w = projective_to_point(get_challenges(TQ_root, M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
    L_at_w = line_function_ext(
        to_extension_field(start_point),
        to_extension_field(output_point),
        w.reshape((1,2,4)) + zeros((width, 1, 1))
    )
    L_at_w_plus_G = line_function_ext(
        to_extension_field(start_point),
        to_extension_field(output_point),
        w_plus_G.reshape((1,2,4)) + zeros((width, 1, 1))
    )
    L2_at_w = line_function_ext(
        to_extension_field(nm1_point),
        to_extension_field(nm2_point),
        w.reshape((1,2,4)) + zeros((width, 1, 1))
    )
    I_at_w = interpolant_ext(
        to_extension_field(start_point),
        to_extension_field(start_state),
        to_extension_field(output_point),
        to_extension_field(output_state),
        w.reshape((1,2,4)) + zeros((width, 1, 1))
    )
    I_at_w_plus_G = interpolant_ext(
        to_extension_field(start_point),
        to_extension_field(start_state),
        to_extension_field(output_point),
        to_extension_field(output_state),
        w_plus_G.reshape((1,2,4)) + zeros((width, 1, 1))
    )
    T_at_w = (extension_field_mul(TQ_at_w, L_at_w) + I_at_w) % M31
    T_at_w_plus_G = (
        extension_field_mul(TQ_at_w_plus_G, L_at_w_plus_G)
        + I_at_w_plus_G
    ) % M31
    C_eval = (
        get_next_state_vector(T_at_w, K_at_w, ext_arith)
        + M31 - T_at_w_plus_G
    ) % M31
    # Z = ext_domain[:,0,np.newaxis]
    # for i in range(1, log2(trace_length)):
    #     Z = (2 * Z**2 + M31 - 1) % M31
    Z_at_w = w[0].reshape((1,4))
    one, _, _ = ext_arith
    for i in range(1, log2(trace_length)):
        Z_at_w = (2 * extension_field_mul(Z_at_w, Z_at_w) + M31 - one) % M31
    computed_H_at_w = extension_field_mul(
        extension_field_mul(C_eval, L2_at_w),
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
        challenges_top * 8 + challenges_bottom
    )
    challenges_next = (challenges+8) % (trace_length*8)
    entropy = TQ_root + b''.join(
        tobytes(H_at_w) for x in
        (TQ_at_w, TQ_at_w_plus_G, H_at_w, H_at_w_plus_G, K_at_w, K_at_w_plus_G)
    )
    fold_factors = (
        get_challenges(TQ_root, M31, (width * 2 + constants_width) * 4)
        .reshape((width * 2 + constants_width, 4))
    )
    L_leaves = line_function(
        start_point,
        output_point,
        append(
            sub_domains[trace_length * 8 + challenges],
            sub_domains[trace_length * 8 + challenges_next],
        )
    )
    I_leaves = interpolant(
        start_point,
        start_state,
        output_point,
        output_state,
        append(
            sub_domains[trace_length * 8 + challenges],
            sub_domains[trace_length * 8 + challenges_next],
        )
    )
    Z_leaves = sub_domains[trace_length * 8 + challenges, 0]
    for i in range(1, log2(trace_length)):
        Z_leaves = (2 * Z_leaves ** 2 + M31 - 1) % M31
    L2_leaves = line_function(
        nm1_point,
        nm2_point,
        sub_domains[trace_length * 8 + challenges]
    )
    inv_L3_leaves = modinv_ext(line_function_ext(
        w,
        w_plus_G,
        sub_domains[trace_length * 8 + challenges]
    ))
    M_at_w = (
        fold_ext(TQ_at_w, fold_factors[:width]) +
        fold_ext(H_at_w, fold_factors[width:width*2]) +
        fold_ext(K_at_w, fold_factors[width*2:])
    ) % M31
    M_at_w_plus_G = (
        fold_ext(TQ_at_w_plus_G, fold_factors[:width]) +
        fold_ext(H_at_w_plus_G, fold_factors[width:width*2]) +
        fold_ext(K_at_w_plus_G, fold_factors[width*2:])
    ) % M31
    I3_leaves = interpolant_ext(
        w,
        M_at_w,
        w_plus_G,
        M_at_w_plus_G,
        sub_domains[trace_length * 8 + challenges]
    )
    T_leaves = (
        TQ_leaves * L_leaves.reshape(L_leaves.shape+(1,))[:NUM_CHALLENGES]
        + I_leaves[:NUM_CHALLENGES]
    ) % M31
    T_next_leaves = (
        TQ_next_leaves * L_leaves.reshape(L_leaves.shape+(1,))[NUM_CHALLENGES:]
        + I_leaves[NUM_CHALLENGES:]
    ) % M31
    C_leaves = (get_next_state_vector(
        T_leaves.swapaxes(0, T_leaves.ndim-1),
        K_leaves.swapaxes(0, T_leaves.ndim-1),
        m31_arith
    ).swapaxes(0, T_leaves.ndim-1) + M31 - T_next_leaves) % M31
    H_leaves = (
        ((C_leaves * L2_leaves.reshape(L2_leaves.shape+(1,))) % M31)
        * modinv(Z_leaves.reshape(Z_leaves.shape+(1,)))
    ) % M31
    merged_leaves = (
        fold(TQ_leaves, fold_factors[:width]) +
        fold(H_leaves, fold_factors[width: width*2]) + 
        fold(K_leaves, fold_factors[width*2:])
    )
    computed_U_leaves = extension_field_mul(
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
