from fast_fft import np, modinv, M31, log2

def mk_junk_data(length):
    log_buffer_size = length.bit_length()
    coeffs = np.ones(2**log_buffer_size, dtype=np.uint64)
    coeffs[1] = 3
    for i in range(1, log_buffer_size):
        coeffs[2**i] = (coeffs[2**(i-1)]**2) % (2**31-19)
    for i in range(1, log_buffer_size):
        coeffs[2**i+1:2**(i+1)] = (coeffs[1:2**i] * coeffs[2**i]) % (2**31-19)
    return coeffs[-length:]

round_constants = mk_junk_data(1536).reshape((64, 24))

def input_to_m31s(data):
    m31s = [
        int.from_bytes(data[i:i+4], 'little')
        for i in range(0, 32, 4)
    ]

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

def hash(in1, in2):
    state = np.zeros(in1.shape[:-1]+(24,), dtype=np.uint64)
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        state += round_constants[i]
        if i >= 4 and i < 60:
            x = state[...,0]
            state[...,0] = (x * x) % M31
            state[...,0] = (state[...,0] * state[...,0]) % M31
            state[...,0] = (state[...,0] * x) % M31
        else:
            x = state
            state = (x * x) % M31
            state = (state * state) % M31
            state = (state * x) % M31

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
