from fast_fft import np, modinv, M31, log2
from fast_arithmetize import m31_arith

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

def poseidon_hash(in1, in2):
    state = np.zeros(in1.shape[:-1]+(24,), dtype=np.uint64)
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        #print('Round {}: {}'.format(i*4, [int(x) for x in state[:3]]))
        state = (state + round_constants[i]) % M31
        if i >= 4 and i < 60:
            x = np.copy(state[...,0])
            state[...,0] = (x * x) % M31
            #print('Round {}: {}'.format(i*4+1, [int(x) for x in state[:3]]))
            state[...,0] = (state[...,0] * state[...,0]) % M31
            #print('Round {}: {}'.format(i*4+2, [int(x) for x in state[:3]]))
            state[...,0] = (state[...,0] * x) % M31
            #print('Round {}: {}'.format(i*4+3, [int(x) for x in state[:3]]))
        else:
            x = state
            state = (x * x) % M31
            #print('Round {}: {}'.format(i*4+1, [int(x) for x in state[:3]]))
            state = (state * state) % M31
            #print('Round {}: {}'.format(i*4+2, [int(x) for x in state[:3]]))
            state = (state * x) % M31
            #print('Round {}: {}'.format(i*4+3, [int(x) for x in state[:3]]))

        state = np.matmul(state, mds) % M31
    return state[...,8:16]

def merkelize(data):
    assert len(data.shape) == 1
    output = np.zeros(data.shape[0] * 2, dtype=np.uint64)
    output[data.shape[0]:] = data
    for i in range(log2(data.shape[0])-1, 2, -1):
        hash_inputs = output[2**(i+1):2**(i+2)].reshape((-1,2,8))
        L, R = hash_inputs[...,0,:], hash_inputs[...,1,:]
        output[2**i: 2**(i+1)] = hash(L, R).reshape((-1,))
    return output

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

def poseidon_next_state(state, c, arith):
    one, add, mul = arith
    L = state[:24]
    R = state[24:]
    Lp = add(L, c[8:])

    new_L = (
        mul(c[0], mul(Lp, Lp)) +
        mul(c[1], mul(L, L)) +
        mul(c[2], mul(L, R)) +
        mul(c[3], np.append(mul(Lp[:1], Lp[:1]), Lp[1:], axis=0)) +
        mul(c[4], np.append(mul(L[:1], L[:1]), L[1:24], axis=0)) +
        mul(c[5], np.append(mul(L[:1], R[:1]), L[1:24], axis=0)) +
        mul(c[6], np.matmul(L.transpose(), mds).transpose() % M31)
    ) % M31
    assert new_L.shape == L.shape
    new_R = (
        mul(c[0], Lp) +
        mul(c[1], R) +
        mul(c[3], Lp) +
        mul(c[4], R)
    ) % M31
    return np.append(new_L, new_R, axis=0)

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

poseidon_constants = np.array(
    OUTER * 4 + INNER * 56 + OUTER * 4,
    dtype=np.uint64
)
poseidon_constants = np.pad(poseidon_constants, ((0,0),(0,24)))
poseidon_constants[::4, 8:] = round_constants

def arith_hash(in1, in2):
    state = np.zeros(48, dtype=np.uint64)
    state[:8] = in1
    state[8:16] = in2
    for i in range(256):
        #print('Round {}: {}'.format(i, [int(x) for x in state[:3]]))
        state = poseidon_next_state(state, poseidon_constants[i], m31_arith)
    return state[8:16]
