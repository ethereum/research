from binary_fields import BinaryFieldElement as B, binmul
from binary_ntt import get_Wi_eval
from utils import log2

MAX_SIZE = 8192

import numpy as np

def build_mul_table():
    table = np.zeros((65536, 65536), dtype=np.uint16)
    
    for i in [2**x for x in range(16)]:
        for j in [2**x for x in range(16)]:
            table[i, j] = binmul(i, j)
        for j in range(3, 65536):
            if j & (j-1):
                top_p_of_2 = 1 << j.bit_length() - 1
                table[i][j] = table[i][top_p_of_2] ^ table[i][j - top_p_of_2]
    
    for i in range(3, 65536):
        if i & (i-1):
            top_p_of_2 = 1 << i.bit_length() - 1
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

def build_big_mul_cache():
    output = np.zeros((8, 8, 8), dtype=np.uint16)
    for i in range(8):
        for j in range(8):
            value = (B(1<<(i*16)) * B(1<<(j*16))).value
            output[i][j] = int_to_bigbin(value)
    return output

big_mul_cache = build_big_mul_cache()
print("Built bigmul cache")

def coerce_to_int(val):
    if isinstance(val, np.ndarray):
        return val
    elif isinstance(val, list):
        return [x if isinstance(x, (int, np.uint16)) else x.value for x in val]
    else:
        return val if isinstance(val, (int, np.uint16)) else val.value

def big_mul(x1, x2):
    assert x1.shape[-1] == x2.shape[-1] == 8
    o = np.zeros(x1.shape, dtype=np.uint16)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                if big_mul_cache[i][j][k] == 1:
                    o[..., k] ^= mul[x1[..., i], x2[..., j]]
                elif big_mul_cache[i][j][k] > 1:
                    o[..., k] ^= mul[
                        big_mul_cache[i][j][k],
                        mul[x1[..., i], x2[..., j]]
                    ]
    return o

def additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    for i in range(log2(vals.size)):
        vals = np.reshape(vals, (1 << i, vals.size >> i))
        halflen = vals.shape[1] >> 1
        start = np.arange(0, vals.size, vals.shape[1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        L = vals[:, :halflen]
        R = vals[:, halflen:]
        sub_input1 = L ^ mul[R, coeff1]
        vals[:, :halflen] = sub_input1
        vals[:, halflen:] = sub_input1 ^ R
    return list(np.reshape(vals, (vals.size, )))

def inv_additive_ntt(vals):
    vals = np.array(vals, dtype=np.uint16)
    for i in range(log2(vals.size)-1, -1, -1):
        vals = np.reshape(vals, (1 << i, vals.size >> i))
        halflen = vals.shape[1] >> 1
        start = np.arange(0, vals.size, vals.shape[1], dtype=np.uint16)
        coeff1 = np.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        coeff2 = coeff1 ^ 1
        L = vals[:, :halflen]
        R = vals[:, halflen:]
        sub_input1 = mul[L, coeff2] ^ mul[R, coeff1]
        sub_input2 = L ^ R
        vals[:, :halflen] = sub_input1
        vals[:, halflen:] = sub_input2
    return list(np.reshape(vals, (vals.size, )))

# Reed-Solomon extension, using the efficient algorithms above
def extend(data, expansion_factor=2):
    data = coerce_to_int(data)
    return [B(int(x)) for x in additive_ntt(
        inv_additive_ntt(data) +
        [0] * len(data) * (expansion_factor - 1)
    )]

def multilinear_poly_eval(evals, pt):
    assert len(evals) == 2 ** len(pt)
    pt = [int_to_bigbin(coerce_to_int(x)) for x in pt]
    evals_as_ints = coerce_to_int(evals)
    if all(v < 2 for v in evals_as_ints):
        evals_array = np.zeros((len(evals), 8), dtype=np.uint16)
        evals_array[:,0] = evals_as_ints
    else:
        evals_array = np.array(
            [int_to_bigbin(x) for x in evals_as_ints]
        )
    for coord in reversed(pt):
        halflen = len(evals_array)//2
        top = evals_array[:halflen]
        bottom = evals_array[halflen:]
        evals_array = big_mul(bottom ^ top, coord) ^ top
    return B(bigbin_to_int(evals_array[0]))

def pack16(vec):
    vec = coerce_to_int(vec)
    o = np.zeros(len(vec)//16, dtype=np.uint16)
    for i in range(15, -1, -1):
        o = o * 2 + vec[i::16]
    return o
