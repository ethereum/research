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
        output[2**i: 2**(i+1)] = hash(L, R).reshape((-1,))
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

def poseidon_next_state(state, c, arith):
    one, add, mul = arith
    L = state[:24]
    R = state[24:]
    Lp = add(L, c[8:])

    newL = np.zeros_like(L)
    newR = np.zeros_like(R)
    multidim = (c.ndim > 1)
    if multidim or c[0]:
        newL += mul(c[0], mul(Lp, Lp))
        newR += mul(c[0], Lp)
    if multidim or c[1]:
        newL += mul(c[1], mul(L, L))
        newR += mul(c[1], R)
    if multidim or c[2]:
        newL += mul(c[2], mul(L, R))
    if multidim or c[3]:
        newL += mul(c[3], append(mul(Lp[:1], Lp[:1]), Lp[1:]))
        newR += mul(c[3], Lp)
    if multidim or c[4]:
        newL += mul(c[4], append(mul(L[:1], L[:1]), L[1:24]))
        newR += mul(c[4], R)
    if multidim or c[5]:
        newL += mul(c[5], append(mul(L[:1], R[:1]), L[1:24]))
    if multidim or c[6]:
        newL += mul(
            c[6],
            _matmul(L.swapaxes(0, L.ndim-1), mds).swapaxes(0, L.ndim-1) % M31
        )

    return append(newL, newR)
    
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

poseidon_constants = np.hstack((
    array(OUTER * 4 + INNER * 56 + OUTER * 4),
    zeros((256, 24))
))
poseidon_constants[::4, 8:] = round_constants

def arith_hash(in1, in2):
    state = zeros(48)
    state[:8] = in1
    state[8:16] = in2
    for i in range(256):
        #print('Round {}: {}'.format(i, [int(x) for x in state[:3]]))
        state = poseidon_next_state(state, poseidon_constants[i], m31_arith)
    return state[8:16]
