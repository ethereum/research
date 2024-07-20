from merkle import merkelize, hash, get_branch, verify_branch
from zorch.m31 import (
    zeros, array, arange, append, tobytes, add, sub, mul, cp as np,
    mul_ext, modinv, modinv_ext, sum as m31_sum, M31, eq, iszero
)
from zorch.m31_circle import Point, ExtendedPoint
M31SQ = M31 ** 2
HALF = 2**30

def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

# Pads an array to a given length
def pad_to(arr, new_len):
    padding = zeros((new_len - arr.shape[0],) + arr.shape[1:])
    return append(arr, padding)

# Confirms that the coefficients respect a given degree bound
def confirm_max_degree(coeffs, bound):
    return iszero(coeffs[bound:])

# Merkelize a list of items
def merkelize_top_dimension(x):
    blob = tobytes(x)
    size = len(blob) // x.shape[0]
    return merkelize([blob[i:i+size] for i in range(0, len(blob), size)])

# Checks if an object needs to be turned into extension-field
# elements, and does so if necessary. object_dim should be the
# dimension of `obj` in the case where it is _not_ extension-field
# elements
def to_ext_if_needed(obj, object_dim):
    if obj.ndim == object_dim:
        return to_extension_field(obj)
    else:
        return obj

def to_extension_field(values):
    o = zeros(values.shape + (4,))
    v = values.reshape(values.shape+(1,))
    o[...,:1] = v
    return o

one_ext = array([1,0,0,0])

# Convert an extension field element to a point in the extension
# field. Guarantees no properties about the point; the use case
# is purely "pick a random point" for Fiat-Shamir-style protocols
def projective_to_point(t):
    t2 = mul_ext(t, t)
    inv_1pt2 = modinv_ext(add(one_ext, t2))
    return ExtendedPoint(
        mul_ext(sub(one_ext, t2), inv_1pt2),
        mul_ext(add(t, t), inv_1pt2)
    )

# Takes as input a N*k array of values, and a vector k extension field
# elements, and computes the size-N linear combination
def fold(vector, fold_factors):
    return m31_sum(
        mul(vector.reshape(vector.shape+(1,)), fold_factors),
        axis=-2
    )

# Same put takes as input a N*k array of extension field elements
def fold_ext(vector, fold_factors):
    return m31_sum(mul_ext(vector, fold_factors), axis=-2)

# Evaluates the simplest polynomial that equals zero across
# the domain of size `degree`, at the given coords. Supports
# the base field or extension field
def eval_zpoly_at(degree, coords, is_extended=False):
    if is_extended:
        Z = coords.x
        one = array([1,0,0,0]).reshape((1,) * (Z.ndim - 1) + (4,))
        _mul = mul_ext
    else:
        Z = coords.x
        one = array([1]).reshape((1,) * Z.ndim)
        _mul = mul
    for i in range(1, log2(degree)):
        Z = sub(2 * _mul(Z, Z) % M31, one)
    return Z

# Converts a list into "reverse bit order", eg:
# 0 1 -> 0 1
# 0 1 2 3 -> 0 2 1 3
# 0 1 2 3 4 5 6 7 -> 0 4 2 6 1 5 3 7
def reverse_bit_order(vals):
    size = vals.shape[0]
    shape_suffix = vals.shape[1:]
    for i in range(log2(size)):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, ::2]
        R = vals[:, 1::2]
        o = np.zeros_like(vals)
        o[:, :half_len] = L
        o[:, half_len:] = R
        vals = o
    return vals.reshape((size,) + shape_suffix)

# Returns the index of a given value in the list created by the
# function above
def rbo_index_to_original(length, index):
    if length == 1:
        return np.zeros_like(index)
    sub = rbo_index_to_original(length >> 1, index >> 1)
    return (1 - (index % 2)) * sub + (index % 2) * (length - 1 - sub)

# Similar to reverse bit order, except when you construct it you
# reverse the right side at each step.
# 0 1 -> 0 1
# 0 1 2 3 -> 0 3 1 2
# 0 1 2 3 4 5 6 7 -> 0 7 3 4 1 6 2 5
# Useful for circle FFTs
def folded_reverse_bit_order(vals):
    vals = np.copy(vals)
    size = vals.shape[0]
    shape_suffix = vals.shape[1:]
    for i in range(log2(size)):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        vals[:, half_len:] = np.flip(vals[:, half_len:], (1,))
    return reverse_bit_order(vals.reshape((size,) + shape_suffix))

# Get challenge indices. Used to choose random linear combinations
# and Merkle branch indices
def get_challenges(entropy, domain_size, num_challenges):
    challenge_data = b''.join(
        hash(entropy + bytes([i//256, i%256])) for i in range((num_challenges + 7) // 8)
    )
    return array([
        int.from_bytes(challenge_data[i:i+4], 'little') % domain_size
        for i in range(0, num_challenges * 4, 4)
    ])

# Generates some pseudorandom numbers mod 2**31
def mk_junk_data(length):
    a = arange(length, length*2)
    return ((3**a) ^ (7**a)) % M31
