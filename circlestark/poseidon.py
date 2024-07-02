from utils import (
    np, modinv, M31, log2, arange, array, zeros, append, m31_arith,
    mk_junk_data, device, pad_to, to_extension_field
)

from arithmetization_builder import (
    example, example_args, generate_constants_table, generate_arguments_table,
    generate_filled_trace, generate_next_state_function,
    get_public_args_indices
)

NUM_HASHES = 151

round_constants = mk_junk_data(1024).reshape((64, 16))

mds44 = array([
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6]
])

mds = zeros((16, 16))
for i in range(0, 16, 4):
    for j in range(0, 16, 4):
        mds[i:i+4, j:j+4] = mds44 * (2 if i==j else 1)

innerdiag = mk_junk_data(16)
mdsinner = zeros((16, 16)) + 1
mdsinner[arange(16), arange(16)] = (innerdiag + 1) % M31

mds_cpu = mds.cpu().numpy()
round_constants_cpu = round_constants.cpu().numpy()
innerdiag_cpu = innerdiag.cpu().numpy()

import numpy

def poseidon_hash(in1, in2):
    state = numpy.zeros(in1.shape[:-1]+(16,), dtype=numpy.int64)
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        if i >= 4 and i < 60:
            state[...,0] = pow5(state[...,0] + round_constants_cpu[i, 0] - M31)
            state = (
                state * innerdiag_cpu
                + numpy.sum(state, axis=-1).reshape(state.shape[:-1]+(1,))
            ) % M31
        else:
            state = numpy.matmul(
                pow5(state + round_constants_cpu[i] - M31),
                mds_cpu
            ) % M31

    return (state[...,8:16] + in2) % M31

def merkelize(data):
    assert len(data.shape) == 1
    output = zeros(data.shape[0] * 2)
    output[data.shape[0]:] = data
    for i in range(log2(data.shape[0])-1, 2, -1):
        hash_inputs = output[2**(i+1):2**(i+2)].reshape((-1,2,8))
        L, R = hash_inputs[...,0,:], hash_inputs[...,1,:]
        output[2**i: 2**(i+1)] = crazy_poseidon(L, R).reshape((-1,))
    return output

# Handle with care. This is only safe if the values in at least one matrix
# are "small" (as floats are only reliable ints up to 2**53)
def _matmul(a, b):
    if hasattr(np, 'tensor'):
        x1 = np.matmul(a.to(np.float64), b.to(np.float64) % 65536).to(np.int64)
        x2 = np.matmul(a.to(np.float64), b.to(np.float64) // 65536).to(np.int64)
        return (x1 + (x2 % M31) * 65536) % M31
    else:
        return np.matmul(a, b) % M31

def pow5(x):
    x2 = (x*x) % M31
    x3 = (x2*x) % M31
    return (x3*x2) % M31

def pow5_arith(x, arith):
    one, add, mul = arith
    x2 = mul(x, x)
    x3 = mul(x, x2)
    return mul(x2, x3)

def vecmul(v1, v2):
    prod = v1 * v2 % M31
    return np.sum(prod, axis=-1) % M31

powers_of_mds = zeros((57, 73, 16))
powers_of_mds[0, 0, 0] = round_constants[4, 0]
powers_of_mds[0, 1:17] = mds
for i in range(56):
    powers_of_mds[i+1] = powers_of_mds[i]
    powers_of_mds[i+1, 17+i, 0] = 1
    powers_of_mds[i+1] = _matmul(powers_of_mds[i+1], mdsinner)
    if i < 55:
        powers_of_mds[i+1, 0, 0] = (
            powers_of_mds[i+1, 0, 0]
            + round_constants[i+5, 0]
        ) % M31

def crazy_poseidon(in1, in2):
    state = zeros(in1.shape[:-1]+(16,)).cpu()
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(4):
        state = pow5(state + round_constants[i] - M31)
        if i < 3:
            state = _matmul(state, mds)
    new_round_values = zeros(in1.shape[:-1]+(73,))
    new_round_values[...,0] = 1
    new_round_values[...,1:17] = state
    for i in range(56):
        m = vecmul(new_round_values[...,:17+i], powers_of_mds[i,:17+i,0])
        new_round_values[...,17+i] = (pow5(m) - m) % M31
    state = _matmul(new_round_values, powers_of_mds[56])
    for i in range(4):
        state = pow5(state + round_constants[60+i] - M31)
        state = _matmul(state, mds)
    return (state[...,8:16] + in2) % M31

def fill_poseidon_trace(hash_inputs, positions):
    N = hash_inputs.shape[0]
    _positions = positions.reshape((N, 1))
    hash_outputs = zeros((N+1, 8))
    for i in range(32):
        L = hash_inputs[i::32] * (1 - _positions[i::32]) + hash_outputs[i:N:32] * _positions[i::32]
        R = hash_inputs[i::32] * _positions[i::32] + hash_outputs[i:N:32] * (1 - _positions[i::32])
        hash_outputs[i+1::32] = np.array(poseidon_hash(L.cpu().numpy(), R.cpu().numpy()))
    trace = zeros((N, 192))
    trace[:, :8] = hash_inputs * (1 - _positions)
    trace[:, 8:16] = hash_inputs * _positions
    trace[1:, :8] += hash_outputs[1:-1] * _positions[1:]
    trace[1:, 8:16] += hash_outputs[1:-1] * (1 - _positions[1:])
    for i in range(4):
        _prev = trace[:,i*16:(i+1)*16]
        _next = pow5(_prev + round_constants[i] - M31)
        if i < 3:
            _next = _matmul(_next, mds)
        trace[:,(i+1)*16:(i+2)*16] = _next
    ones = zeros((hash_inputs.shape[0], 1)) + 1
    for i in range(56):
        m = vecmul(
            np.hstack((ones, trace[:,64:80+i])),
            powers_of_mds[i,:17+i,0]
        )
        trace[:,80+i] = (pow5(m) - m) % M31

    for i in range(4):
        if i == 0:
            _prev = _matmul(
                np.hstack((ones, trace[:,64:136])),
                powers_of_mds[56]
            )
        else:
            _prev = trace[:,120+16*i:136+16*i]
        _next = pow5((_prev + round_constants[60+i]) % M31)
        _next = _matmul(_next, mds)
        if i < 3:
            trace[:,136+16*i:152+16*i] = _next
        else:
            trace[:,184:192] = (_next[:,8:16] + trace[:, 8:16]) % M31
    return trace

def poseidon_constraint_check(state, next_state, constants, arith):
    one, add, mul = arith
    o = zeros((184,) + state.shape[1:])
    depth = state.ndim - one.ndim - 1

    def fix_rc_row(rc_row):
        rc_row = rc_row.reshape(rc_row.shape + (1,)*depth)
        if one.ndim == 1:
            return to_extension_field(rc_row)
        return rc_row

    for i in range(4):
        _prev = state[i*16:(i+1)*16]
        rc = fix_rc_row(round_constants[i])
        expected = pow5_arith((_prev + rc) % M31, arith)
        if i < 3:
            expected = _matmul(expected.swapaxes(0,-1), mds).swapaxes(0,-1)
        o[i*16:(i+1)*16] = state[(i+1)*16:(i+2)*16] - expected
    ones = np.zeros_like(o[:1])
    if one.ndim == 1:
        ones[...,0] = 1
    else:
        ones += 1
    for i in range(56):
        m = vecmul(
            append(ones, state[64:80+i]).swapaxes(0,-1),
            powers_of_mds[i,:17+i,0]
        ).swapaxes(0,-1)
        o[64+i] = state[80+i] - (pow5_arith(m, arith) - m)
    for i in range(4):
        if i==0:
            _prev = _matmul(
                append(ones, state[64:136]).swapaxes(0,-1),
                powers_of_mds[56]
            ).swapaxes(0,-1)
        else:
            _prev = state[120+16*i:136+16*i]
        rc = fix_rc_row(round_constants[60+i])
        expected = pow5_arith((_prev + rc) % M31, arith)
        expected = _matmul(expected.swapaxes(0,-1), mds).swapaxes(0,-1)
        if i < 3:
            o[120+16*i:136+16*i] = state[136+16*i:152+16*i] - expected
        else:
            o[168:176] = state[184:192] - (expected[8:16] + state[8:16])
    o[176:184] = mul(
        (one - constants[0]) % M31,
        mul(
            (next_state[:8] - state[184:192]) % M31,
            (next_state[8:16] - state[184:192]) % M31
        )
    )
    return o % M31
