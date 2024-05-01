from binary_fields import BinaryFieldElement as B, binmul
from binary_ntt import get_Wi_eval
from utils import log2

MAX_SIZE = 8192

import numpy

def build_mul_table():
    table = numpy.zeros((65536, 65536), dtype=numpy.uint16)
    
    for i in range(16):
        table[1<<i] = [binmul(1<<i, j) for j in range(65536)]
    
    for i in range(3, 65536):
        if i & (i-1):
            top_power_of_two = 1 << i.bit_length() - 1
            table[i] = table[top_power_of_two] ^ table[i - top_power_of_two]
    
    return table

mul = build_mul_table()
print("Built multiplication table")

assert mul[12345, 23456] == (B(12345) * B(23456)).value

def build_inv_table():
    output = numpy.ones(65536, dtype=numpy.uint16)
    exponents = numpy.arange(0, 65536, 1, dtype=numpy.uint16)
    for i in range(15):
        exponents = mul[exponents, exponents]
        output = mul[exponents, output]
    return output

inv = build_inv_table()
print("Built inversion table")

assert mul[7890, inv[7890]] == 1

def build_Wi_eval_cache():
    # Maintains a cache of Wi(pt) values, so Wi_eval_cache[dim][pt] = W{dim}(pt)
    Wi_eval_cache = numpy.zeros((log2(MAX_SIZE), MAX_SIZE), dtype=numpy.uint16)

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

assert Wi_eval_cache[7, 383] == 229

def additive_ntt(vals):
    vals = numpy.array(vals, dtype=numpy.uint16)
    for i in range(log2(vals.size)):
        vals = numpy.reshape(vals, (1 << i, vals.size >> i))
        halflen = vals.shape[1] >> 1
        start = numpy.arange(0, vals.size, vals.shape[1], dtype=numpy.uint16)
        coeff1 = numpy.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        L = vals[:, :halflen]
        R = vals[:, halflen:]
        sub_input1 = L ^ mul[R, coeff1]
        vals[:, :halflen] = sub_input1
        vals[:, halflen:] = sub_input1 ^ R
    return list(numpy.reshape(vals, (vals.size, )))

def inv_additive_ntt(vals):
    vals = numpy.array(vals, dtype=numpy.uint16)
    for i in range(log2(vals.size)-1, -1, -1):
        vals = numpy.reshape(vals, (1 << i, vals.size >> i))
        halflen = vals.shape[1] >> 1
        start = numpy.arange(0, vals.size, vals.shape[1], dtype=numpy.uint16)
        coeff1 = numpy.reshape(Wi_eval_cache[log2(halflen), start], (start.size, 1))
        coeff2 = coeff1 ^ 1
        L = vals[:, :halflen]
        R = vals[:, halflen:]
        sub_input1 = mul[L, coeff2] ^ mul[R, coeff1]
        sub_input2 = L ^ R
        vals[:, :halflen] = sub_input1
        vals[:, halflen:] = sub_input2
    return list(numpy.reshape(vals, (vals.size, )))

# Reed-Solomon extension, using the efficient algorithms above
def extend(data, expansion_factor=2):
    data = [B(val).value for val in data]
    return [B(int(x)) for x in additive_ntt(
        inv_additive_ntt(data) +
        [0] * len(data) * (expansion_factor - 1)
    )]
