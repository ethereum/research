from fast_fft import (
    np, modinv, M31, M31SQ, log2, fft, inv_fft, sub_domains,
    bary_eval, point_add, to_extension_field,
    zeros, array, arange, tobytes, append, to_ext_if_needed
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

def line_function(P1, P2, domain, arith):
    one, add, mul = arith
    a = (P2[1] + M31 - P1[1]) % M31
    b = (P1[0] + M31 - P2[0]) % M31
    c = (M31SQ + mul(P2[0], P1[1]) - mul(P1[0], P2[1])) % M31
    if one.ndim == 1:
        domain = to_ext_if_needed(domain, object_dim=2)
        P1 = to_ext_if_needed(P1, object_dim=1)
        P2 = to_ext_if_needed(P2, object_dim=1)

    return (mul(a, domain[:,0]) + mul(b, domain[:,1]) + c) % M31

def interpolant(P1, v1, P2, v2, domain, arith):
    one, add, mul = arith
    depth = len(v1.shape) - len(one.shape)
    inv = modinv_ext if one.ndim == 1 else modinv
    v1 = v1.reshape((1,)+v1.shape)
    v2 = v2.reshape((1,)+v2.shape)
    if one.ndim == 1:
        domain = to_ext_if_needed(domain, object_dim=2)
        P1 = to_ext_if_needed(P1, object_dim=1)
        P2 = to_ext_if_needed(P2, object_dim=1)
    if np.array_equal(P1[0], P2[0]):
        y = domain[:,1].reshape((domain.shape[0],) + (1,) * depth + one.shape)
        slope = mul((v2 - v1) % M31, inv((P2[1] - P1[1]) % M31))
        return (v1 + mul((y - P1[1]) % M31, slope)) % M31
    else:
        x = domain[:,0].reshape((domain.shape[0],) + (1,) * depth + one.shape)
        slope = mul((v2 - v1) % M31, inv((P2[0] - P1[0]) % M31))
        return (v1 + mul((x - P1[0]) % M31, slope)) % M31

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
    array(1),
    lambda *x: sum(x) % M31,
    lambda x,y: x*y % M31
)
ext_arith = (
    one,
    lambda *x: sum(x) % M31,
    extension_field_mul
)

def public_args_to_vanish_and_interp(domain_size,
                                     indices,
                                     vals,
                                     arith,
                                     out_domain=None):
    one, add, mul = arith
    inv = modinv_ext if one.ndim == 1 else modinv
    assert len(indices) % 2 == 0
    next_power_of_2 = 2**(len(indices)-1).bit_length() * 2
    assert next_power_of_2 < domain_size
    depth = len(vals.shape) - 1 - one.ndim
    lines = []
    eval_domain = sub_domains[next_power_of_2: next_power_of_2*2]
    if out_domain is not None:
        eval_domain = append(eval_domain, out_domain)
    vpoly = one.reshape((1,)+one.shape)+zeros((eval_domain.shape[0],)+one.shape)
    points = sub_domains[domain_size + array(indices)]
    if one.ndim == 1:
        points = to_ext_if_needed(points, object_dim=2)
    for i in range(0, len(indices), 2):
        lines.append(
            line_function(points[i], points[i+1], eval_domain, arith)
        )
        vpoly = mul(vpoly, lines[-1])
    interp = zeros((eval_domain.shape[0],) + vals.shape[1:])
    for i in range(0, len(indices), 2):
        vpoly_adjusted = (
            mul(vpoly, inv(lines[i//2]))
            .reshape((eval_domain.shape[0],) + (1,) * depth + one.shape)
        )
        y1 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i], arith)
        y2 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i+1], arith)
        I = interpolant(
            points[i], mul(vals[i], inv(y1)),
            points[i+1], mul(vals[i+1], inv(y2)),
            eval_domain,
            arith
        )
        interp += mul(vpoly_adjusted, I)
    if out_domain is not None:
        return vpoly[next_power_of_2:], interp[next_power_of_2:] % M31
    else:
        return vpoly, interp % M31

def eval_zpoly_at(degree, coords):
    Z = coords[...,0]
    for i in range(1, log2(degree)):
        Z = (2 * Z**2 - 1) % M31
    return Z

def eval_zpoly_at_ext(degree, coords):
    Z = coords[...,0,:]
    for i in range(1, log2(degree)):
        Z = (2 * extension_field_mul(Z, Z) - one) % M31
    return Z

def mk_stark(get_next_state_vector, trace_width, constants, arguments, public_args=tuple()):
    import time
    rounds, constants_width = constants.shape[:2]
    trace_length = 2**(rounds+1).bit_length()
    print('Trace length: {}'.format(trace_length))
    trace = zeros((trace_length, trace_width))
    START = time.time()
    constants = pad_to(constants, trace_length)
    arguments = pad_to(arguments, trace_length)
    arguments_width = arguments.shape[1]
    for i in range(rounds+1):
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
    # Extend to 8x
    ext8_domain = sub_domains[trace_length*8: trace_length*16]
    C_ext8 = inv_fft(pad_to(fft(C_ext4), trace_length*8))
    assert confirm_max_degree(fft(C_ext8), trace_length*3)
    trace_ext8 = inv_fft(pad_to(trace_coeffs, trace_length*8))
    constants_ext8 = inv_fft(pad_to(constants_coeffs, trace_length*8))
    arguments_ext8 = inv_fft(pad_to(arguments_coeffs, trace_length*8))
    Va, Ia = public_args_to_vanish_and_interp(
        trace_length,
        public_args,
        arguments[array(public_args)],
        m31_arith
    )
    Va = Va.reshape((Va.shape[0], 1))
    Ia_ext8 = inv_fft(pad_to(fft(Ia), trace_length*8))
    Va_ext8 = inv_fft(pad_to(fft(Va), trace_length*8))
    args_quotient_ext8 = (arguments_ext8 - Ia_ext8) * modinv(Va_ext8) % M31
    assert confirm_max_degree(fft(args_quotient_ext8), trace_length)
    Vt, It = public_args_to_vanish_and_interp(
        trace_length,
        (0, rounds),
        trace[array((0, rounds))],
        m31_arith
    )
    Vt = Vt.reshape((Vt.shape[0], 1))
    It_ext8 = inv_fft(pad_to(fft(It), trace_length*8))
    Vt_ext8 = inv_fft(pad_to(fft(Vt), trace_length*8))
    trace_quotient_ext8 = (trace_ext8 - It_ext8) * modinv(Vt_ext8) % M31
    assert confirm_max_degree(fft(trace_quotient_ext8), trace_length)
    print('Generated size-8x polynomials', time.time() - START)
    Z_ext8 = eval_zpoly_at(
        trace_length,
        ext8_domain.reshape((trace_length*8, 1, 2))
    )
    H_ext8 = C_ext8 * modinv(Z_ext8) % M31
    H_coeffs = fft(H_ext8)
    assert confirm_max_degree(H_coeffs, trace_length*2+3)
    print('About to make trees!', time.time() - START)
    stack_ext8 = np.hstack((
        trace_quotient_ext8,
        args_quotient_ext8,
        constants_ext8,
        H_ext8
    ))
    TA_ext8 = stack_ext8[:,:trace_width + arguments_width]
    TA_tree = merkelize_top_dimension(TA_ext8)
    print('Generated first tree!', time.time() - START)
    K_tree = merkelize_top_dimension(constants_ext8)
    print('Generated second tree!', time.time() - START)
    # Fold and prove
    G = sub_domains[trace_length//2]
    bump = to_extension_field(sub_domains[trace_length*8])
    w = projective_to_point(get_challenges(TA_tree[1], M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
    #print('T_at_w', bary_eval(to_extension_field(trace_ext8), w, ext_arith)[:3])
    #print('K_at_w', bary_eval(to_extension_field(constants_ext8), w, ext_arith)[:3])
    #print('A_at_w', bary_eval(to_extension_field(arguments_ext8), w, ext_arith)[:3])
    #print('T_at_w_plus_G', bary_eval(to_extension_field(trace_ext8), w_plus_G, ext_arith)[:3])
    #print('C_at_w', bary_eval(to_extension_field(C_ext8), w, ext_arith)[:3])
    w_bump = extension_point_add(w, bump)
    comb = np.hstack((
        stack_ext8[::2],
        append(stack_ext8[8::2],stack_ext8[:8:2]),
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
        fold(stack_ext8, fold_factors)
    ) % M31
    merged_poly_coeffs = fft(merged_poly)
    assert confirm_max_degree(merged_poly_coeffs, trace_length * 3 + 3)
    print('Generated merged poly!', time.time() - START)
    L3 = line_function(w, w_plus_G, ext8_domain, ext_arith)
    I3 = interpolant(
        w,
        fold_ext(S_at_w, fold_factors),
        w_plus_G,
        fold_ext(S_at_w_plus_G, fold_factors),
        ext8_domain,
        ext_arith
    )
    assert np.array_equal(bary_eval(I3, w_plus_G, ext_arith), fold_ext(S_at_w_plus_G, fold_factors))
    master_quotient = extension_field_mul(
        (merged_poly - I3) % M31, modinv_ext(L3)
    )
    print('Generated master_quotient!', time.time() - START)
    master_quotient_coeffs = fft(master_quotient)
    assert confirm_max_degree(master_quotient_coeffs, trace_length * 3)

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
        "TA_root": TA_tree[1],
        "TA_branches": [get_branch(TA_tree, c) for c in challenges],
        "TA_leaves": TA_ext8[challenges],
        "TA_next_branches": [get_branch(TA_tree, c) for c in challenges_next],
        "TA_next_leaves": TA_ext8[challenges_next],
        "K_root": K_tree[1],
        "K_branches": [get_branch(K_tree, c) for c in challenges],
        "K_leaves": constants_ext8[challenges],
        "S_at_w": S_at_w,
        "S_at_w_plus_G": S_at_w_plus_G,
    }

def get_vk(trace_width, constants, arguments_width, public_args_positions):
    rounds = constants.shape[0]
    trace_length = 2**rounds.bit_length()
    constants_coeffs = fft(pad_to(constants, trace_length))
    constants_ext = inv_fft(pad_to(constants_coeffs, trace_length*8))
    return {
        "root": merkelize_top_dimension(constants_ext)[1],
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
    start_point = sub_domains[trace_length]
    output_point = sub_domains[trace_length + rounds]
    nm1_point = sub_domains[trace_length*2-1]
    nm2_point = sub_domains[trace_length*2-2]
    output_state = proof["output_state"]
    TA_root = proof["TA_root"]
    w = projective_to_point(get_challenges(TA_root, M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
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
        extension_field_mul(
            S_at_w[:trace_width],
            bary_eval(to_extension_field(Vt), w, ext_arith)
        )
        + bary_eval(to_extension_field(It), w, ext_arith)
    ) % M31
    T_at_w_plus_G = (
        extension_field_mul(
            S_at_w_plus_G[:trace_width],
            bary_eval(to_extension_field(Vt), w_plus_G, ext_arith)
        )
        + bary_eval(to_extension_field(It), w_plus_G, ext_arith)
    ) % M31
    A_at_w = (
        extension_field_mul(
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
    Z_at_w = w[0].reshape((1,4))
    one, _, _ = ext_arith
    for i in range(1, log2(trace_length)):
        Z_at_w = (2 * extension_field_mul(Z_at_w, Z_at_w) + M31 - one) % M31
    computed_H_at_w = extension_field_mul(
        C_at_w,
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
        out_domain=sub_domains[trace_length * 8 + challenges]
    )
    Vt_leaves, It_leaves = public_args_to_vanish_and_interp(
        trace_length,
        (0, rounds),
        np.stack((zeros(trace_width), output_state)),
        m31_arith,
        out_domain=append(
            sub_domains[trace_length * 8 + challenges],
            sub_domains[trace_length * 8 + challenges_next],
        )
    )
    Z_leaves = sub_domains[trace_length * 8 + challenges, 0]
    for i in range(1, log2(trace_length)):
        Z_leaves = (2 * Z_leaves ** 2 + M31 - 1) % M31
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
        sub_domains[trace_length * 8 + challenges],
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
        sub_domains[trace_length * 8 + challenges],
        ext_arith
    )

    merged_leaves = (
        fold(np.hstack((TA_leaves, K_leaves, H_leaves)), fold_factors)
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
            TA_root, ci, tobytes(TA_leaves[i]), TA_branches[i])
        assert verify_branch(
            TA_root, nci, tobytes(TA_next_leaves[i]), TA_next_branches[i])
        assert verify_branch(
            K_root, ci, tobytes(K_leaves[i]), K_branches[i])
    return True
