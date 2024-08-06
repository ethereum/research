from merkle import merkelize, hash, get_branch, verify_branch
from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like
)
import cupy as cp
HALF = 2**30

def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

# Pads an array to a given length
def pad_to(arr, new_len):
    padding = arr.__class__.zeros((new_len - arr.shape[0],) + arr.shape[1:])
    return arr.__class__.append(arr, padding)

# Confirms that the coefficients respect a given degree bound
def confirm_max_degree(coeffs, bound):
    return coeffs[bound:] == 0

# Merkelize a list of items
def merkelize_top_dimension(x):
    blob = x.tobytes()
    size = len(blob) // x.shape[0]
    return merkelize([blob[i:i+size] for i in range(0, len(blob), size)])

# Convert an extension field element to a point in the extension
# field. Guarantees no properties about the point; the use case
# is purely "pick a random point" for Fiat-Shamir-style protocols
def projective_to_point(t):
    t2 = t**2
    inv_1pt2 = 1 / (t2 + 1)
    return Point(
        (1 - t2) * inv_1pt2,
        t * 2 * inv_1pt2
    )

# Takes as input a N*k array of values, and a vector k extension field
# elements, and computes the size-N linear combination
def fold(vector, fold_factors):
    return ExtendedM31.sum(
        vector * fold_factors,
        axis=-1
    )

# Evaluates the simplest polynomial that equals zero across
# the domain of size `degree`, at the given coords. Supports
# the base field or extension field
def eval_zpoly_at(degree, coords):
    Z = coords.x
    for i in range(1, log2(degree)):
        Z = 2 * Z * Z - 1
    return Z

# Converts a list into "reverse bit order", eg:
# 0 1 -> 0 1
# 0 1 2 3 -> 0 2 1 3
# 0 1 2 3 4 5 6 7 -> 0 4 2 6 1 5 3 7
def reverse_bit_order(vals):
    size = vals.shape[0]
    shape_suffix = vals.shape[1:]
    for i in range(log2(size)):
        vals = vals.reshape((1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, ::2]
        R = vals[:, 1::2]
        o = zeros_like(vals)
        o[:, :half_len] = L
        o[:, half_len:] = R
        vals = o
    return vals.reshape((size,) + shape_suffix)

# Returns the index of a given value in the list created by the
# function above
def rbo_index_to_original(length, index, first_round=True):
    assert index.__class__ == cp.ndarray
    if length == 1:
        return cp.zeros_like(index)
    sub = rbo_index_to_original(length >> 1, index >> 1, False)
    if first_round:
        return (1 - (index % 2)) * sub*2 + (index % 2) * (length - 1 - sub*2)
    else:
        return (1 - (index % 2)) * sub + (index % 2) * (length//2 + sub)

# Similar to reverse bit order, except when you construct it you
# reverse the right side at each step.
# 0 1 -> 0 1
# 0 1 2 3 -> 0 3 1 2
# 0 1 2 3 4 5 6 7 -> 0 7 3 4 1 6 2 5
# Useful for circle FFTs
def folded_reverse_bit_order(vals):
    o = zeros_like(vals)
    o[::2] = reverse_bit_order(vals[::2])
    o[1::2] = reverse_bit_order(vals[1::2][::-1])
    return o

# Get challenge indices. Used to choose random linear combinations
# and Merkle branch indices
def get_challenges(entropy, domain_size, num_challenges):
    challenge_data = b''.join(
        hash(entropy + bytes([i//256, i%256])) for i in range((num_challenges + 7) // 8)
    )
    return cp.array([
        int.from_bytes(challenge_data[i:i+4], 'little') % domain_size
        for i in range(0, num_challenges * 4, 4)
    ], dtype=cp.uint32)

# Generates some pseudorandom numbers mod 2**31
def mk_junk_data(length):
    a = cp.arange(length, length*2, dtype=cp.uint32)
    return M31(((3**a) ^ (7**a)) % modulus)
