from fast_fft import (
    np, modinv, M31, log2, fft, inv_fft, sub_domains
)

def mk_junk_data(length):
    a = np.arange(length, length*2, dtype=np.uint64)
    return ((3**a) ^ (7**a)) % M31

round_constants = mk_junk_data(1536).reshape((64, 24))

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

def mk_stark(get_next_state_vector, start_state, constants):
    rounds = constants.shape[0]
    trace_length = 2**rounds.bit_length()
    trace = np.zeros((trace_length,) + start_state.shape)
    trace[0] = start_state
    for i in range(rounds):
        trace[i+1] = get_next_state_vector(trace[i], constants[i])
    end_state = trace[rounds]
    trace_coeffs = fft(trace)
    trace_ext = inv_fft(pad_to(trace_coeffs, trace_length*4))
    constants_coeffs = fft(pad_to(constants, trace_length))
    constants_ext = inv_fft(pad_to(constants_coeffs, trace_length*4))
    # Trace must satisfy
    #   C(T[x], T[x+1]) * (X - p0)
    # + r * (T(x) - I) / L[p0, pe]
    # = Z(x) * H(x)
    C = get_next_state_vector(trace_ext, constants) - np.append(trace_ext[1:], trace_ext[:1])
    C_coeffs = fft(C * (
        sub_domains[trace_length*4:trace_length*8,0]
        - sub_domains[trace_length]
    ))
    print(np.sum(C_coeffs, axis=1))
