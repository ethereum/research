from fast_fft import (
    np, modinv, M31, log2, arange, array, zeros, append
)
from fast_arithmetize import m31_arith

def mk_junk_data(length):
    a = arange(length, length*2)
    return ((3**a) ^ (7**a)) % M31

round_constants = mk_junk_data(1536).reshape((64, 24))

mds44 = array([
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6]
])

mds = zeros((24, 24))
for i in range(0, 24, 4):
    for j in range(0, 24, 4):
        mds[i:i+4, j:j+4] = mds44 * (2 if i==j else 1)

def poseidon_hash(in1, in2):
    state = zeros(in1.shape[:-1]+(24,))
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

        state = _matmul(state, mds) % M31
    return state[...,8:16]

def merkelize(data):
    assert len(data.shape) == 1
    output = zeros(data.shape[0] * 2)
    output[data.shape[0]:] = data
    for i in range(log2(data.shape[0])-1, 2, -1):
        hash_inputs = output[2**(i+1):2**(i+2)].reshape((-1,2,8))
        L, R = hash_inputs[...,0,:], hash_inputs[...,1,:]
        output[2**i: 2**(i+1)] = poseidon_hash(L, R).reshape((-1,))
    return output

# Handle with care. This is only safe if the values in at least one matrix
# are "small" (as floats are only reliable ints up to 2**53)
def _matmul(a, b):
    if hasattr(np, 'tensor'):
        return np.matmul(a.to(np.float64), b.to(np.float64)).to(np.int64)
    else:
        return np.matmul(a, b)
    #small = np.matmul((a&65535).to(np.float32), b.to(np.float32)).to(np.int64)
    #large = np.matmul((a>>16).to(np.float32), b.to(np.float32)).to(np.int64)
    #return small + (large << 16)

# 0: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[0...23]
# 1: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[24...47]
# 2: state[0...23] = state_prev[0...23] * state_prev[24...47], state_prev[24...47] = 0
# 3: state[0] = state_prev[0]**2, state[24] = state_prev[0]
# 4: state[0] = state_prev[0]**2, state[24] = state_prev[24]
# 5: state[0] = state_prev[0] * state_prev[24]
# 6: state[0...23] = MDS * state_prev[0...23], state_prev[24...47] = 0

# N(P(x) = P(xw))
# C(P(xw), P(x)) = N(P(x)) - P(xw) = 0

def poseidon_next_state(state, c, a, arith):
    one, add, mul = arith
    L = state[:24]
    R = state[24:]

    if c.ndim > 1 or np.count_nonzero(c[:8]) > 1:
        Z24 = np.zeros_like(R)
        Z8 = np.zeros_like(R[:8])
        Lp = (L + c[8:]) % M31
        MAT = _matmul(L.swapaxes(0, L.ndim-1), mds).swapaxes(0, L.ndim-1) % M31
        return (
            mul(c[0], append(mul(Lp, Lp), Lp)) +
            mul(c[1], append(mul(L, L), R)) +
            mul(c[2], append(mul(L, R), Z24)) +
            mul(c[3], append(mul(Lp[:1], Lp[:1]), Lp[1:], Lp)) +
            mul(c[4], append(mul(L[:1], L[:1]), L[1:24], R)) +
            mul(c[5], append(mul(L[:1], R[:1]), L[1:24], Z24)) +
            mul(c[6], append(mul(c[6], MAT), Z24)) +
            mul(c[7], (
                mul((one - a[8]) % M31, append(a[:8], L[8:16], Z8, Z24)) +
                mul(a[8], append(L[8:16], a[:8], Z8, Z24))
            ) % M31)
        ) % M31
    else:
        if c[0]:
            Lp = (L + c[8:]) % M31
            return append(mul(Lp, Lp), Lp)
        elif c[1]:
            return append(mul(L, L), R)
        elif c[2]:
            return append(mul(L, R), zeros(24))
        elif c[3]:
            Lp = (L + c[8:]) % M31
            return append(mul(Lp[:1], Lp[:1]), Lp[1:], Lp)
        elif c[4]:
            return append(mul(L[:1], L[:1]), L[1:24], R)
        elif c[5]:
            return append(mul(L[:1], R[:1]), L[1:24], zeros(24))
        elif c[6]:
            MAT = _matmul(L, mds) % M31
            return append(mul(c[6], MAT), zeros(24))
        elif c[7]:
            Z24 = np.zeros_like(R)
            Z8 = np.zeros_like(R[:8])
            return (
                mul((one - a[8]) % M31, append(a[:8], L[8:16], Z8, Z24)) +
                mul(a[8], append(L[8:16], a[:8], Z8, Z24))
            ) % M31
        else:
            return zeros(48)
    
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

INITIAL = [
    [0,0,0,0,0,0,0,1]
]

COMPONENT = INITIAL + (OUTER * 4 + INNER * 56 + OUTER * 4)

NUM_HASHES = 51

poseidon_constants = np.hstack((
    array(INITIAL + COMPONENT * NUM_HASHES),
    zeros((1 + 257 * NUM_HASHES, 24))
))
for i in range(NUM_HASHES):
    poseidon_constants[257*i+2:257*(i+1)+1:4, 8:] = round_constants

def arith_hash(in1, in2):
    state = zeros(48)
    arguments = zeros((258, 9))
    arguments[0, :8] = in1
    arguments[0, 8] = 1
    arguments[1, :8] = in2
    arguments[1, 8] = 1
    for i in range(258):
        #print('Round {}: {}'.format(i, [int(x) for x in state[:3]]))
        state = poseidon_next_state(
            state,
            poseidon_constants[i],
            arguments[i],
            m31_arith
        )
    return state[8:16]
