from binary_fields import BinaryFieldElement as B, binmul
from binary_ntt import get_Wi_eval
from utils import log2

MAX_SIZE = 8192

import numpy as np

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

def build_Wi_eval_cache():
    # Maintains a cache of Wi(pt) values, so Wi_eval_cache[dim][pt] = W{dim}(pt)
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

def int_to_bigbin(value):
    return np.array(
        [(value >> (k*16)) & 65535 for k in range(8)],
        dtype=np.uint16
    )

def bigbin_to_int(value):
    return sum(int(x) << (16*i) for i,x in enumerate(value))

def coerce_to_int(val):
    if isinstance(val, np.ndarray):
        return val
    elif isinstance(val, list):
        return [x if isinstance(x, (int, np.uint16)) else x.value for x in val]
    else:
        return val if isinstance(val, (int, np.uint16)) else val.value

def big_mul(x1, x2):
    N = x1.shape[-1]
    if N == 1:
        return mul[x1, x2]
    L1, L2 = x1[..., :N//2], x2[..., :N//2]
    R1, R2 = x1[..., N//2:], x2[..., N//2:]
    L1L2 = big_mul(L1, L2)
    R1R2 = big_mul(R1, R2)
    cofactor = np.zeros(N//2, dtype=np.uint16)
    if N >= 4:
        cofactor[N//4] = 1
    else:
        cofactor[0] = 256
    R1R2_high = big_mul(R1R2, cofactor)
    Z3 = big_mul(L1 ^ R1, L2 ^ R2)
    o = np.concatenate((
        L1L2 ^ R1R2,
        (Z3 ^ L1L2 ^ R1R2 ^ R1R2_high)
    ), axis=-1)
    return o

def additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    shape_prefix = vals.shape[:-1]
    size = vals.shape[-1]
    for i in range(log2(size)):
        vals = np.reshape(vals, shape_prefix + (1 << i, size >> i))
        halflen = vals.shape[-1] >> 1
        start = np.arange(0, size, vals.shape[-1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        L = vals[..., :, :halflen]
        R = vals[..., :, halflen:]
        sub_input1 = L ^ mul[R, coeff1]
        vals[..., :, :halflen] = sub_input1
        vals[..., :, halflen:] = sub_input1 ^ R
    return np.reshape(vals, shape_prefix + (size,))

def inv_additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    shape_prefix = vals.shape[:-1]
    size = vals.shape[-1]
    for i in range(log2(size)-1, -1, -1):
        vals = np.reshape(vals, shape_prefix + (1 << i, size >> i))
        halflen = vals.shape[-1] >> 1
        start = np.arange(0, size, vals.shape[-1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        coeff2 = coeff1 ^ 1
        L = vals[..., :, :halflen]
        R = vals[..., :, halflen:]
        sub_input1 = mul[L, coeff2] ^ mul[R, coeff1]
        sub_input2 = L ^ R
        vals[..., :, :halflen] = sub_input1
        vals[..., :, halflen:] = sub_input2
    return np.reshape(vals, shape_prefix + (size,))

# Reed-Solomon extension, using the efficient algorithms above
def extend(data, expansion_factor=2):
    new_dimension = data.shape[-1] * (expansion_factor - 1)
    return additive_ntt(np.pad(
        inv_additive_ntt(data),
        ([0]*len(data.shape), [0]*(len(data.shape)-1)+[new_dimension])
    ))

def bytestobits(data):
    big_end = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype('<u2')
    return np.fliplr(big_end.reshape(-1, 8)).ravel()

def uint16s_to_bits(data):
    big_end = np.unpackbits(data.astype('<u2').view(np.uint8), axis=-1)
    return np.fliplr(big_end.reshape(-1, 8)).reshape(big_end.shape)

def bitstobytes(data):
    return np.packbits(np.fliplr(data.reshape(-1,8)).ravel()).tobytes()

def bits_to_uint16s(data):
    bit_flipped = np.fliplr(data.reshape(-1,8)).reshape(data.shape)
    return np.packbits(bit_flipped, axis=-1).view('<u2')

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

def pack16(vec):
    vec = coerce_to_int(vec)
    o = np.zeros(len(vec)//16, dtype=np.uint16)
    for i in range(15, -1, -1):
        o = o * 2 + vec[i::16]
    return o

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
