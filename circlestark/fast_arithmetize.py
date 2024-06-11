from fast_fft import (
    np, modinv, M31, M31SQ, log2, fft, inv_fft, sub_domains,
    bary_eval, point_add, to_extension_field
)
from fast_fri import (
    get_challenges, extension_field_mul, extension_point_add,
    modinv_ext, prove_low_degree
)
from merkle import merkelize, hash, get_branch, verify_branch

def mk_junk_data(length):
    a = np.arange(length, length*2, dtype=np.uint64)
    return ((3**a) ^ (7**a)) % M31

round_constants = mk_junk_data(1536).reshape((64, 24))

NUM_CHALLENGES = 80

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
    return (a * domain[:,0,np.newaxis] + b * domain[:,1,np.newaxis] + c) % M31

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
    return (
        v1 +
        extension_field_mul(
            to_extension_field(domain_coords) + M31 - coord1,
            slope
        )
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

def mk_stark(get_next_state_vector, start_state, constants):
    rounds = constants.shape[0]
    trace_length = 2**rounds.bit_length()
    print('Trace length: {}'.format(trace_length))
    trace = np.zeros((trace_length,) + start_state.shape, dtype=np.uint64)
    trace[0] = start_state
    for i in range(rounds):
        trace[i+1] = get_next_state_vector(trace[i], constants[i])
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
    print('Extended trace shape:', trace_ext.shape)
    C = np.zeros_like(trace_ext)
    rolled_trace = np.roll(trace_ext, -8)
    for i in range(trace_length * 8):
        C[i] = (
            get_next_state_vector(trace_ext[i], constants_ext[i])
            + M31 - rolled_trace[i]
        ) % M31
    #C = (get_next_state_vector(trace_ext, constants_ext) + M31 - np.roll(trace_ext, -8)) % M31
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
    # Sanity check
    baryL = bary_eval(L, sub_domains[9999])
    baryI = bary_eval(I, sub_domains[9999])
    baryLplus = bary_eval(L, point_add(sub_domains[9999], G))
    baryIplus = bary_eval(I, point_add(sub_domains[9999], G))
    assert ((
        get_next_state_vector(
            (bary_eval(T_quotient, sub_domains[9999]) * baryL + baryI)[0] % M31,
            bary_eval(constants_ext, sub_domains[9999])
        ) + M31 - (
            bary_eval(T_quotient, point_add(sub_domains[9999], G)) * baryLplus + baryIplus
        ) % M31
    ) % M31) * (
        bary_eval(ext_domain[:,0], sub_domains[9999]) + M31 - start_point[0]
    ) % M31 == (
        bary_eval(H, sub_domains[9999])
        * bary_eval(Z, sub_domains[9999])
    ) % M31
    T_tree = merkelize(chunkify(T_quotient))
    fold_factors = (
        get_challenges(T_tree[1], M31, 4 * start_state.size)
        .reshape((start_state.size, 4))
    )
    # Fold and prove
    H_folded = np.sum(H[...,np.newaxis] * fold_factors % M31, axis=-2) % M31
    w = projective_to_point(get_challenges(T_tree[1], M31, 4))
    w_plus_G = extension_point_add(w, to_extension_field(G))

    Lw = line_function_ext(w, w_plus_G, ext_domain)
    Iw = interpolant_ext(w, bary_eval(H_folded, w), w_plus_G, bary_eval(H_folded, w_plus_G), ext_domain)
    H_deep = extension_field_mul(
        (H_folded + M31 - Iw) % M31,
        modinv_ext(Lw)
    )
    folded_fri = prove_low_degree(H_deep)
    entropy = b''.join(folded_fri["roots"] + [x.astype(np.uint32).tobytes() for x in folded_fri["final_values"]])
    challenges = get_challenges(
        entropy, H_deep.shape[0], NUM_CHALLENGES
    )
    branches = [get_branch(T_tree, c) for c in challenges]

    return folded_fri, branches
