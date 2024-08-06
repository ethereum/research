from utils import (
    log2, mk_junk_data, reverse_bit_order, cp
)

from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like, Z, G, matmul
)
from precomputes import invx, invy, sub_domains, rbos

round_constants = mk_junk_data(1024).reshape((64, 16))

mds44 = M31([
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6]
])

mds = M31.zeros((16, 16))
for i in range(0, 16, 4):
    for j in range(0, 16, 4):
        mds[i:i+4, j:j+4] = mds44 * (2 if i==j else 1)

innerdiag = mk_junk_data(16)
mdsinner = M31.zeros((16, 16)) + 1
mdsinner[cp.arange(16), cp.arange(16)] = innerdiag + 1

import numpy

mds_cpu = mds.value.get()
round_constants_cpu = round_constants.value.get()
innerdiag_cpu = innerdiag.value.get()

def pow5_cpu(x):
    x2 = (x*x) % modulus
    x3 = (x2*x) % modulus
    return (x3*x2) % modulus

def poseidon_hash_cpu(in1, in2):
    state = numpy.zeros(in1.shape[:-1]+(16,), dtype=numpy.int64)
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        if i >= 4 and i < 60:
            state[...,0] = pow5_cpu(
                state[...,0] + round_constants_cpu[i, 0] - modulus
            )
            state = (
                state * innerdiag_cpu
                + numpy.sum(state, axis=-1).reshape(state.shape[:-1]+(1,))
            ) % modulus
        else:
            state = numpy.matmul(
                pow5_cpu(state + round_constants_cpu[i] - modulus),
                mds_cpu
            ) % modulus

    return (state[...,8:16] + in2) % modulus

def mul_by_mds(data):
    return matmul(data, mds, assume_second_input_small=True)

def poseidon_hash(in1, in2):
    state = zeros(in1.shape[:-1]+(16,))
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        if i >= 4 and i < 60:
            state[...,0] = pow5(add(state[...,0], round_constants[i, 0]))
            state = add(
                mul(state, innerdiag),
                m31_sum(state, axis=-1).reshape(state.shape[:-1]+(1,))
            )
        else:
            mul_input = pow5(add(state, round_constants[i]))
            state = mul_by_mds(mul_input)

    return add(state[...,8:16], in2)

# We're proving a STARK of a series of Merkle branches, each 32 long
def fill_poseidon_trace(hash_inputs, positions):
    N = hash_inputs.shape[0]
    positions = M31(positions.reshape((N, 1)))
    hash_outputs = M31.zeros((N+1, 8))
    # Build all N/32 branches in parallel to make things faster
    for i in range(32):
        L = (
            hash_inputs[i::32] * (1 - positions[i::32])
            + hash_outputs[i:N:32] * positions[i::32]
        )
        R = (
            hash_inputs[i::32] * positions[i::32]
            + hash_outputs[i:N:32] * (1 - positions[i::32])
        )
        # Do the hashing in numpy. It's slower per byte, but
        # less overhead, and even at 8192 hashes, overhead of
        # pinging back and forth between computer and GPU is
        # the more important thing to optimize
        #hash_outputs[i+1::32] = poseidon_hash(L, R)
        hash_outputs[i+1::32] = M31(cp.array(
            poseidon_hash_cpu(
                L.value.get().astype(numpy.uint64),
                R.value.get().astype(numpy.uint64)
            ).astype(numpy.uint32),
            dtype=cp.uint32
        ))
    # Trace has 192 columns
    trace = M31.zeros((N, 192))
    # First 16 are the input
    trace[:, :8] = hash_inputs * (1 - positions)
    trace[:, 8:16] = hash_inputs * positions
    trace[1:, :8] += hash_outputs[1:-1] * positions[1:]
    trace[1:, 8:16] += hash_outputs[1:-1] * (1 - positions[1:])
    # First four rounds (minus the last matmul)
    for i in range(4):
        _prev = trace[:,i*16:(i+1)*16]
        _next = (_prev + round_constants[i]) ** 5
        if i < 3:
            _next = mul_by_mds(_next)
        trace[:,(i+1)*16:(i+2)*16] = _next
    # Middle 56 rounds, only first cell is edited. We use clever
    # tricks to only use one trace cell per round
    ones = M31.zeros((hash_inputs.shape[0], 1)) + 1
    for i in range(56):
        m = vecmul(
            M31.append(ones, trace[:,64:80+i], axis=1),
            powers_of_mds[i,:17+i,0]
        )
        trace[:,80+i] = (m**5 - m)

    # Last four rounds (plus the matmul of the previous round)
    for i in range(4):
        if i == 0:
            _prev = matmul(
                M31.append(ones, trace[:,64:136], axis=1),
                powers_of_mds[56]
            )
        else:
            _prev = trace[:,120+16*i:136+16*i]
        _next = (_prev + round_constants[60+i]) ** 5
        _next = mul_by_mds(_next)
        if i < 3:
            trace[:,136+16*i:152+16*i] = _next
        else:
            # In the last round, only save 8 cells, we don't need the
            # others
            trace[:,184:192] = _next[:,8:16] + trace[:, 8:16]
    # assert eq(trace[1,184:], poseidon_hash(trace[1, :8], trace[1, 8:16]))
    return trace

mdsinner = M31.zeros((16, 16)) + 1
mdsinner[cp.arange(16), cp.arange(16)] = (innerdiag + 1)

powers_of_mds = M31.zeros((57, 73, 16))
powers_of_mds[0, 0, 0] = round_constants[4, 0]
powers_of_mds[0, 1:17] = mds
for i in range(56):
    powers_of_mds[i+1] = powers_of_mds[i]
    powers_of_mds[i+1, 17+i, 0] = 1
    powers_of_mds[i+1] = matmul(powers_of_mds[i+1], mdsinner)
    if i < 55:
        powers_of_mds[i+1, 0, 0] += round_constants[i+5, 0]

def vecmul(v1, v2):
    return m31.sum(v1 * v2, axis=-1)

vecmul = matmul

state_mask = M31(
    [[1 if i<j+16 else 0 for i in range(72)] for j in range(56)]
)

# Run the constraint function C(T(x), T(next(x)), K(x))
def poseidon_constraint_check(state, next_state, constants):
    cls = state.__class__
    o = cls.zeros((184,) + state.shape[1:])
    ones = zeros_like(o[:1]) + 1
    depth = state.ndim - 1

    rc = round_constants.reshape(round_constants.shape + (1,)*depth)

    # First four rounds (minus the last matmul)
    for i in range(4):
        _prev = state[i*16:(i+1)*16]
        expected = (_prev + rc[i]) ** 5
        if i < 3:
            expected = mul_by_mds(expected.swapaxes(0,-1)).swapaxes(0,-1)
        o[i*16:(i+1)*16] = state[(i+1)*16:(i+2)*16] - expected

    # Middle 56 rounds
    states = cls.zeros((56,73) + state.shape[1:])
    states[:,1:] = state[64:136]
    states[:,1:] *= state_mask.reshape((56,72) + (1,) * (state.ndim-1))
    states[:,0] = ones
    m2 = cls.sum(
        states * powers_of_mds[:56,:,0].reshape((56,73)+(1,)*(state.ndim-1)),
        axis=1
    )
    if m2.ndim > 1:
        m2 = m2.swapaxes(1,-1)
    o[64:120] = state[80:136] - (m2 ** 5 - m2)

    # Last 4 rounds, plus the last matmul of the previous round
    for i in range(4):
        if i==0:
            prev = matmul(
                cls.append(ones, state[64:136]).swapaxes(0,-1),
                powers_of_mds[56]
            ).swapaxes(0,-1)
        else:
            prev = state[120+16*i:136+16*i]
        expected = (prev + rc[60+i]) ** 5
        expected = mul_by_mds(expected.swapaxes(0,-1)).swapaxes(0,-1)
        if i < 3:
            o[120+16*i:136+16*i] = state[136+16*i:152+16*i] - expected
        else:
            o[168:176] = state[184:192] - (expected[8:16] + state[8:16])

    # Checking consistency between the output and the next input. One of
    # the two next input leaves must be the output leaf
    o[176:184] = (
        (1 - constants[0])
        * (next_state[:8] - state[184:192])
        * (next_state[8:16] - state[184:192])
    )
    return o
