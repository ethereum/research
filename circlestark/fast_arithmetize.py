from fast_fft import (
    np, modinv, M31, M31SQ, log2, fft, inv_fft, sub_domains,
    bary_eval, point_add, to_extension_field, bary_eval_ext
)
from fast_fri import (
    get_challenges, extension_field_mul, extension_point_add,
    modinv_ext, prove_low_degree, verify_low_degree,
    folded_reverse_bit_order, rbo_index_to_original
)
from merkle import merkelize, hash, get_branch, verify_branch

def mk_junk_data(length):
    a = np.arange(length, length*2, dtype=np.uint64)
    return ((3**a) ^ (7**a)) % M31

round_constants = mk_junk_data(1536).reshape((64, 24))

NUM_CHALLENGES = 80
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND

mds44 = np.array([
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6]
], dtype=np.uint64)

mds = np.zeros((24, 24), dtype=np.uint64)
for i in range(0, 24, 4):
    for j in range(0, 24, 4):
        mds[i:i+4, j:j+4] = mds44 * (2 if i==j else 1)

# 0: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[0...23]
# 1: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[24...47]
# 2: state[0...23] = state_prev[0...23] * state_prev[24...47], state_prev[24...47] = 0
# 3: state[0] = state_prev[0]**2, state[24] = state_prev[0]
# 4: state[0] = state_prev[0]**2, state[24] = state_prev[24]
# 5: state[0] = state_prev[0] * state_prev[24]
# 6: state[0...23] = MDS * state_prev[0...23], state_prev[24...47] = 0

# N(P(x) = P(xw))
# C(P(xw), P(x)) = N(P(x)) - P(xw) = 0

def get_next_state_vector(state, c):
    L = state[...,:24]
    R = state[...,24:]
    L_plus_rc = (L + c[...,8:]) % M31

    new_L = (
        c[...,0] * (L_plus_rc ** 2) +
        c[...,1] * (L ** 2) +
        c[...,2] * (L * R) +
        c[...,3] * np.append(L_plus_rc[:1] ** 2, L_plus_rc[1:], axis=0) +
        c[...,4] * np.append(L[:1] ** 2, L[1:24], axis=0) +
        c[...,5] * np.append(L[:1] * R[:1], L[1:24], axis=0) +
        c[...,6] * np.matmul(L, mds)
    ) % M31
    new_R = (
        c[...,0] * L_plus_rc +
        c[...,1] * R +
        c[...,3] * L_plus_rc +
        c[...,4] * R
    )
    return np.append(new_L, new_R, axis=-1)

OUTER = [
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0]
]

INNER = [
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0]
]

constants_vector = np.array(
    OUTER * 4 + INNER * 56 + OUTER * 4,
    dtype=np.uint64
)
constants_vector = np.pad(constants_vector, ((0,0),(0,24)))
constants_vector[::4, 8:] = round_constants

def arith_hash(in1, in2):
    state = np.zeros(48, dtype=np.uint64)
    state[:8] = in1
    state[8:16] = in2
    for i in range(256):
        #print('Round {}: {}'.format(i, [int(x) for x in state[:3]]))
        state = get_next_state_vector(state, constants_vector[i])
    return state[8:16]

def pad_to(arr, new_len):
    return np.pad(
        arr,
        ((0,new_len - arr.shape[0]),) + ((0,0),) * (len(arr.shape) - 1)
    )

def confirm_max_degree(coeffs, bound):
    return np.array_equal(coeffs[bound:], np.zeros_like(coeffs[bound:]))

def line_function(P1, P2, domain):
    denominator = (P1[0] * P2[1] + M31SQ - P1[1] * P2[0]) % M31
    a = (P2[1] + M31 - P1[1]) * modinv(denominator) % M31
    b = (P1[0] + M31 - P2[0]) * modinv(denominator) % M31
    c = (M31SQ * 2 - (a * P1[0] + b * P1[1])) % M31
    return (a * domain[:,0] + b * domain[:,1] + c) % M31

def line_function_ext(P1, P2, domain):
    denominator = (
        extension_field_mul(P1[0], P2[1])
        + M31 - extension_field_mul(P1[1], P2[0])
    ) % M31
    inv_denominator = modinv_ext(denominator)
    a = extension_field_mul((P2[1] + M31 - P1[1]) % M31, inv_denominator)
    b = extension_field_mul((P1[0] + M31 - P2[0]) % M31, inv_denominator)
    c = (M31 * 2 - (
        extension_field_mul(a, P1[0]) + extension_field_mul(b, P1[1])
    )) % M31
    if len(domain.shape) == 2:
        return (a * domain[:,0,np.newaxis] + b * domain[:,1,np.newaxis] + c) % M31
    else:
        return (
            extension_field_mul(a, domain[:,0])
            + extension_field_mul(b, domain[:,1])
            + c
        ) % M31

def interpolant(P1, v1, P2, v2, domain):
    if P1[0] == P2[0]:
        slope = (v2 + M31 - v1) * modinv(P2[1] + M31 - P1[1]) % M31
        return v1 + (domain[:,1] + M31 - P1[1]) * slope % M31
    else:
        slope = (v2 + M31 - v1) * modinv(P2[0] + M31 - P1[0]) % M31
        return v1 + (domain[:,0] + M31 - P1[0]) * slope % M31

def interpolant_ext(P1, v1, P2, v2, domain):
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
    one = np.array([1,0,0,0], dtype=np.uint64)
    inv_1pt2 = modinv_ext((one + t2) % M31)
    return np.array([
        extension_field_mul((one + M31 - t2) % M31, inv_1pt2),
        extension_field_mul((2 * t) % M31, inv_1pt2)
    ], dtype=np.uint64)

def mulM31(x, y):
    return x * y % M31

def fold(vector, fold_factors):
    return np.sum(vector[...,np.newaxis] * fold_factors % M31, axis=-2) % M31

def fold_ext(vector, fold_factors):
    return np.sum(extension_field_mul(vector, fold_factors), axis=-2) % M31

def mk_stark(get_next_state_vector, start_state, constants):
    rounds, constants_width = constants.shape[:2]
    width = start_state.shape[0]
    trace_length = 2**rounds.bit_length()
    print('Trace length: {}'.format(trace_length))
    trace = np.zeros((trace_length,) + start_state.shape, dtype=np.uint64)
    trace[0] = start_state
    for i in range(rounds):
        trace[i+1] = get_next_state_vector(trace[i], constants[i], mulM31)
    output_state = trace[rounds]
    trace_coeffs = fft(trace)
    trace_ext = inv_fft(pad_to(trace_coeffs, trace_length*8))
    constants_coeffs = fft(pad_to(constants, trace_length))
    constants_ext = inv_fft(pad_to(constants_coeffs, trace_length*8))
    assert confirm_max_degree(fft(constants_ext), trace_length)
    assert confirm_max_degree(fft(trace_ext), trace_length)
    ext_domain = sub_domains[trace_length*8: trace_length*16]
    start_point = sub_domains[trace_length]
    output_point = sub_domains[trace_length + rounds]
    G = sub_domains[trace_length//2]
    # Trace must satisfy
    #   C(T[x], T[x+G]) * (X - p0)
    # = Z(x) * H(x)
    C = np.zeros_like(trace_ext)
    rolled_trace = np.roll(trace_ext, -8)
    for i in range(trace_length * 8):
        C[i] = (
            get_next_state_vector(trace_ext[i], constants_ext[i], mulM31)
            + M31 - rolled_trace[i]
        ) % M31
    assert confirm_max_degree(fft(np.roll(trace_ext, -8)), trace_length)
    assert confirm_max_degree(fft(C), trace_length*3+1)
    C_exempt_start = C * (
        ext_domain[:,0,np.newaxis]
        + M31 - start_point[0]
    ) % M31
    C_coeffs = fft(C_exempt_start)
    assert confirm_max_degree(C_coeffs, trace_length*3+2)
    Z = ext_domain[:,0,np.newaxis]
    for i in range(1, log2(trace_length)):
        Z = (2 * Z**2 + M31 - 1) % M31
    H = C_exempt_start * modinv(Z) % M31
    H_coeffs = fft(H)
    assert confirm_max_degree(H_coeffs, trace_length*2+2)
    I = interpolant(
        start_point, start_state,
        output_point, output_state,
        ext_domain
    )[:,np.newaxis]
    L = line_function(start_point, output_point, ext_domain)[:,np.newaxis]
    T_quotient = ((trace_ext + M31 - I) * modinv(L)) % M31
    assert confirm_max_degree(fft(T_quotient), trace_length)
    T_tree = merkelize([x.tobytes() for x in T_quotient.astype(np.uint32)])
    K_tree = merkelize([x.tobytes() for x in constants_ext.astype(np.uint32)])
    # Fold and prove
    w = projective_to_point(get_challenges(T_tree[1], M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
    H_at_w = bary_eval_ext(to_extension_field(H), w)
    H_at_w_plus_G = bary_eval_ext(to_extension_field(H), w_plus_G)
    TQ_at_w = bary_eval_ext(to_extension_field(T_quotient), w)
    TQ_at_w_plus_G = bary_eval_ext(to_extension_field(T_quotient), w_plus_G)
    K_at_w = bary_eval_ext(to_extension_field(constants_ext), w)
    K_at_w_plus_G = bary_eval_ext(to_extension_field(constants_ext), w_plus_G)
    entropy = T_tree[1] + b''.join(
        H_at_w.astype(np.uint32).tobytes() for x in
        (TQ_at_w, TQ_at_w_plus_G, H_at_w, H_at_w_plus_G, K_at_w, K_at_w_plus_G)
    )
    fold_factors = (
        get_challenges(T_tree[1], M31, (width * 2 + constants_width) * 4)
        .reshape((width * 2 + constants_width, 4))
    )
    merged_poly = (
        fold_ext(to_extension_field(T_quotient), fold_factors[:width]) + 
        fold_ext(to_extension_field(H), fold_factors[width: width*2]) +
        fold_ext(to_extension_field(constants_ext), fold_factors[width*2:])
    ) % M31
    L3 = line_function_ext(w, w_plus_G, ext_domain)
    I3 = interpolant_ext(
        w,
        bary_eval_ext(merged_poly, w),
        w_plus_G,
        bary_eval_ext(merged_poly, w_plus_G),
        ext_domain
    )
    master_quotient = extension_field_mul(
        (merged_poly + M31 - I3) % M31,
        modinv_ext(L3)
    )

    fri_proof = prove_low_degree(master_quotient)
    entropy = (b''.join(
        fri_proof["roots"] +
        [x.astype(np.uint32).tobytes() for x in fri_proof["final_values"]]
    ))
    challenges_raw = get_challenges(
        entropy, trace_length*8, NUM_CHALLENGES
    )
    fri_top_leaf_count = trace_length*8 >> FOLDS_PER_ROUND
    challenges_top = challenges_raw % fri_top_leaf_count
    challenges_bottom = challenges_raw // fri_top_leaf_count
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
        "root": merkelize(
            [x.tobytes() for x in constants_ext.astype(np.uint32)]
        )[1],
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
    trace_length = len_evaluations // 8
    G = sub_domains[trace_length//2]
    start_point = sub_domains[trace_length]
    output_point = sub_domains[trace_length + rounds]
    output_state = proof["output_state"]
    TQ_root = proof["TQ_root"]
    w = projective_to_point(get_challenges(TQ_root, M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))
    L_at_w, L_at_w_plus_G = line_function_ext(
        to_extension_field(start_point),
        to_extension_field(output_point),
        np.array([w, w_plus_G])
    )
    I_at_w, I_at_w_plus_G = interpolant_ext(
        to_extension_field(start_point),
        to_extension_field(start_state),
        to_extension_field(output_point),
        to_extension_field(output_state),
        np.array([w, w_plus_G])
    )
    C_eval = (
        get_next_state_vector(
            (extension_field_mul(TQ_at_w, L_at_w) + I_at_w)[0] % M31,
            K_at_w,
            extension_field_mul
        ) + M31 - (
            extension_field_mul(TQ_at_w_plus_G, L_at_w_plus_G) + I_at_w_plus_G
        ) % M31
    ) % M31
    # Z = ext_domain[:,0,np.newaxis]
    # for i in range(1, log2(trace_length)):
    #     Z = (2 * Z**2 + M31 - 1) % M31
    Z_at_w = np.array([w[0]])
    one = np.array([1,0,0,0], dtype=np.uint64)
    for i in range(1, log2(trace_length)):
        Z_at_w = (2 * extension_field_mul(Z_at_w, Z_at_w) + M31 - one) % M31
    L2_eval = (
        w[0] + M31 - to_extension_field(start_point[0])
    ) % M31
    assert np.array_equal(
        extension_field_mul(
            extension_field_mul(C_eval, L2_eval),
            modinv_ext(Z_at_w)
        ),
        H_at_w
    )
    entropy = b''.join(
        fri_proof["roots"] +
        [x.astype(np.uint32).tobytes() for x in fri_proof["final_values"]]
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
        H_at_w.astype(np.uint32).tobytes() for x in
        (TQ_at_w, TQ_at_w_plus_G, H_at_w, H_at_w_plus_G, K_at_w, K_at_w_plus_G)
    )
    fold_factors = (
        get_challenges(TQ_root, M31, (width * 2 + constants_width) * 4)
        .reshape((width * 2 + constants_width, 4))
    )
    L_leaves = line_function(
        start_point,
        output_point,
        np.append(
            sub_domains[trace_length * 8 + challenges],
            sub_domains[trace_length * 8 + challenges_next],
            axis=0
        )
    )
    I_leaves = interpolant(
        start_point,
        start_state,
        output_point,
        output_state,
        np.append(
            sub_domains[trace_length * 8 + challenges],
            sub_domains[trace_length * 8 + challenges_next],
            axis=0
        )
    )
    Z_leaves = sub_domains[trace_length * 8 + challenges, 0]
    one = np.array([1,0,0,0], dtype=np.uint64)
    for i in range(1, log2(trace_length)):
        Z_leaves = (2 * Z_leaves ** 2 + M31 - 1) % M31
    L2_leaves = (
        sub_domains[trace_length * 8 + challenges, 0]
        + M31 - start_point[0]
    ) % M31
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

    for i in range(NUM_CHALLENGES):
        TQ_leaf = TQ_leaves[i]
        TQ_next_leaf = TQ_next_leaves[i]
        K_leaf = K_leaves[i]
        U_leaf = fri_proof["leaf_values"][0][i, challenges_bottom[i]]
        T_leaf = (TQ_leaf * L_leaves[i] + I_leaves[i]) % M31
        T_next_leaf = (
            TQ_next_leaf * L_leaves[i + NUM_CHALLENGES]
            + I_leaves[i + NUM_CHALLENGES]
        ) % M31
        C_leaf = (
            get_next_state_vector(T_leaf, K_leaf, mulM31)
            + M31 - T_next_leaf
        ) % M31
        H_leaf = ((C_leaf * L2_leaves[i]) % M31) * modinv(Z_leaves[i]) % M31
        merged_leaf = (
            fold(TQ_leaf, fold_factors[:width]) +
            fold(H_leaf, fold_factors[width: width*2]) + 
            fold(K_leaf, fold_factors[width*2:])
        ) % M31
        computed_U_leaf = extension_field_mul(
            (merged_leaf + M31 - I3_leaves[i]) % M31,
            inv_L3_leaves[i]
        )
        assert np.array_equal(computed_U_leaf, U_leaf)

    def b(x):
        return x.astype(np.uint32).tobytes()

    for i in range(NUM_CHALLENGES):
        ci, nci = int(challenges[i]), int(challenges_next[i])
        assert verify_branch(
            TQ_root, ci, b(TQ_leaves[i]), TQ_branches[i])
        assert verify_branch(
            TQ_root, nci, b(TQ_next_leaves[i]), TQ_next_branches[i])
        assert verify_branch(
            K_root, ci, b(K_leaves[i]), K_branches[i])
