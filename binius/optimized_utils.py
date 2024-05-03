from binary_fields import BinaryFieldElement as B, binmul
from binary_ntt import get_Wi_eval
from utils import log2

MAX_SIZE = 8192

import numpy as np

# Build a 65536*65536 multiplication table of uint16's (takes 8 GB RAM)
def build_mul_table():
    table = np.zeros((65536, 65536), dtype=np.uint16)
    
    for i in [2**x for x in range(16)]:
        top_p_of_2 = 0
        for j in range(1, 65536):
            if (j & (j-1)) == 0:
                table[i, j] = binmul(i, j)
                top_p_of_2 = j
            else:
                table[i][j] = table[i][top_p_of_2] ^ table[i][j - top_p_of_2]
    
    for i in range(1, 65536):
        if (i & (i-1)) == 0:
            top_p_of_2 = i
        else:
            table[i] = table[top_p_of_2] ^ table[i - top_p_of_2]
    
    return table

mul = build_mul_table()
print("Built multiplication table")

assert mul[12345, 23456] == (B(12345) * B(23456)).value

# Build a table mapping x -> 1/x
def build_inv_table():
    output = np.ones(65536, dtype=np.uint16)
    exponents = np.arange(0, 65536, 1, dtype=np.uint16)
    for i in range(15):
        exponents = mul[exponents, exponents]
        output = mul[exponents, output]
    return output

inv = build_inv_table()
print("Built inversion table")

assert mul[7890, inv[7890]] == 1

# Build a cache of Wi(pt) values, so Wi_eval_cache[dim][pt] = W{dim}(pt)
def build_Wi_eval_cache():
    Wi_eval_cache = np.zeros((log2(MAX_SIZE), MAX_SIZE), dtype=np.uint16)

    Wi_eval_cache[0] = list(range(MAX_SIZE))

    for dim in range(1, log2(MAX_SIZE)):
        prev = Wi_eval_cache[dim-1]
        prev_quot = Wi_eval_cache[dim-1][1<<dim]
        inv_quot = inv[mul[prev_quot, prev_quot ^ 1]]
        Wi_eval_cache[dim] = (
            mul[mul[prev, prev^1], inv_quot]
        )
    return Wi_eval_cache

Wi_eval_cache = build_Wi_eval_cache()
print("Built Wi cache")

# Convert a 128-bit integer into the field element representation we
# use here, which is a length-8 vector of uint16's
def int_to_bigbin(value):
    return np.array(
        [(value >> (k*16)) & 65535 for k in range(8)],
        dtype=np.uint16
    )

# Convert a uint16-representation big field element into an int
def bigbin_to_int(value):
    return sum(int(x) << (16*i) for i,x in enumerate(value))

# Multiplying an element in the i'th level subfield by X_i can be done in
# an optimized way. See sec III of https://ieeexplore.ieee.org/document/612935
def mul_by_Xi(x, N):
    assert x.shape[-1] == N
    if N == 1:
        return mul[x, 256]
    L, R = x[..., :N//2], x[..., N//2:]
    outR = mul_by_Xi(R, N//2) ^ L
    return np.concatenate((R, outR), axis=-1)

# Multiplies together two field elements, using the Karatsuba algorithm
def big_mul(x1, x2):
    N = x1.shape[-1]
    if N == 1:
        return mul[x1, x2]
    L1, L2 = x1[..., :N//2], x2[..., :N//2]
    R1, R2 = x1[..., N//2:], x2[..., N//2:]
    L1L2 = big_mul(L1, L2)
    R1R2 = big_mul(R1, R2)
    R1R2_high = mul_by_Xi(R1R2, N//2)
    Z3 = big_mul(L1 ^ R1, L2 ^ R2)
    o = np.concatenate((
        L1L2 ^ R1R2,
        (Z3 ^ L1L2 ^ R1R2 ^ R1R2_high)
    ), axis=-1)
    return o

# Converts a polynomial with coefficients in the special basis, into evaluations
# See page 4 of https://arxiv.org/pdf/1802.03932
def additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    shape_prefix = vals.shape[:-1]
    size = vals.shape[-1]
    for i in range(log2(size)):
        vals = np.reshape(vals, shape_prefix + (1 << i, size >> i))
        halflen = vals.shape[-1] >> 1
        start = np.arange(0, size, vals.shape[-1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (-1, 1))
        L = vals[..., :, :halflen]
        R = vals[..., :, halflen:]
        sub_input1 = L ^ mul[R, coeff1]
        vals[..., :, :halflen] = sub_input1
        vals[..., :, halflen:] = sub_input1 ^ R
    return np.reshape(vals, shape_prefix + (size,))

# Converts evaluations into coefficients
def inv_additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    shape_prefix = vals.shape[:-1]
    size = vals.shape[-1]
    for i in range(log2(size)-1, -1, -1):
        vals = np.reshape(vals, shape_prefix + (1 << i, size >> i))
        halflen = vals.shape[-1] >> 1
        start = np.arange(0, size, vals.shape[-1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (-1, 1))
        coeff2 = coeff1 ^ 1
        L = vals[..., :, :halflen]
        R = vals[..., :, halflen:]
        sub_input1 = mul[L, coeff2] ^ mul[R, coeff1]
        sub_input2 = L ^ R
        vals[..., :, :halflen] = sub_input1
        vals[..., :, halflen:] = sub_input2
    return np.reshape(vals, shape_prefix + (size,))

# Reed-Solomon extension, using the binary-FFT algorithms above
def extend(data, expansion_factor=2):
    new_dimension = data.shape[-1] * (expansion_factor - 1)
    return additive_ntt(np.pad(
        inv_additive_ntt(data),
        ([0]*len(data.shape), [0]*(len(data.shape)-1)+[new_dimension])
    ))

# Converts n bytes into n*8 bits
def bytestobits(data):
    big_end = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype('<u2')
    return np.fliplr(big_end.reshape(-1, 8)).ravel()

# Converts n uint16's into n*16 bits. Works over arrays of uint16s
def uint16s_to_bits(data):
    big_end = np.unpackbits(data.astype('<u2').view(np.uint8), axis=-1)
    return np.fliplr(big_end.reshape(-1, 8)).reshape(big_end.shape)

# Inverse of bytestobits
def bitstobytes(data):
    return np.packbits(np.fliplr(data.reshape(-1,8)).ravel()).tobytes()

# Inverse of uint16s_to_bits
def bits_to_uint16s(data):
    bit_flipped = np.fliplr(data.reshape(-1,8)).reshape(data.shape)
    return np.packbits(bit_flipped, axis=-1).view('<u2')

# Treat `evals` as the evaluations of a multilinear polynomial over {0,1}^k.
# That is, if evals is [a,b,c,d], then a=P(0,0), b=P(1,0), c=P(0,1), d=P(1,1)
def multilinear_poly_eval(data, pt):
    if isinstance(pt, list):
        pt = np.array([
            int_to_bigbin(x) if isinstance(x, int) else x for x in pt
        ])
    assert log2(len(data) * 8) == pt.shape[0]
    evals_array = np.zeros((len(data) * 8, 8), dtype=np.uint16)
    evals_array[:,0] = bytestobits(data)
    for coord in reversed(pt):
        halflen = len(evals_array)//2
        top = evals_array[:halflen]
        bottom = evals_array[halflen:]
        evals_array = big_mul(bottom ^ top, coord) ^ top
    return evals_array[0]

# Returns the 2^k-long list of all possible results of walking through pt
# (an evaluation point) and at each step taking either coord or 1-coord.
# This is a natural companion method to `multilinear_poly_eval`, because
# it gives a list where `output[i]` equals
# `multilinear_poly_eval([0, 0 ... 1 ... 0, 0], pt)`, where the 1 is in
# position i.
def evaluation_tensor_product(pt):
    if isinstance(pt, list):
        pt = np.array([
            int_to_bigbin(x) if isinstance(x, int) else x for x in pt
        ])
    o = np.array([int_to_bigbin(1)])
    for coord in pt:
        o_times_coord = big_mul(o, coord)
        o = np.concatenate((
            o_times_coord ^ o,
            o_times_coord
        ))
    return o

# Given a list of N objects, and a list of length-N bitvectors representing
# subsets of those objects, compute the xor-sum of each subset. Uses the main
# subroutine of Pippenger-style algorithms, see: https://ethresear.ch/t/7238
def multisubset(values, bits):
    assert values.shape[0] == bits.shape[-1]
    GROUPING = 4
    subsets = np.zeros(
        (values.shape[0] // GROUPING, 16) + values.shape[1:],
        dtype=values.dtype
    )
    for i in range(GROUPING):
        subsets[:,1<<i,...] = values[i::4]
    top_p_of_2 = 2
    for i in range(3, 1<<GROUPING):
        if (i & (i-1)) == 0:
            top_p_of_2 = i
        else:
            subsets[:,i,...] = (
                subsets[:,top_p_of_2,...] ^
                subsets[:,i - top_p_of_2,...]
            )
    bits_GROUPING_at_a_time = bits.reshape(bits.shape[:-1] + (-1, GROUPING))
    index_columns = np.sum( 
        bits_GROUPING_at_a_time * [1<<i for i in range(GROUPING)],
        axis=-1
    )       
    o = np.bitwise_xor.reduce(
        subsets[np.arange(index_columns.shape[-1]), index_columns],
        axis=len(bits.shape)-1
    )
    return o
