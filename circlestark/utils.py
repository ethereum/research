from merkle import merkelize, hash, get_branch, verify_branch
import torch as np
np.array = np.tensor
np.array_equal = np.equal
np.copy = np.clone
device = np.device("cuda" if np.cuda.is_available() else "cpu")

M31 = 2**31-1
M31SQ = (2**31-1)**2
HALF = 2**30

# Converts a list to an array (which benefits from GPU acceleration)
def array(x):
    return np.array(x, dtype=np.int64, device=device)

# Creates an array of zeros of the given shape
def zeros(shape):
    return np.zeros(shape, dtype=np.int64, device=device)

# Converts an array to bytes
def tobytes(shape):
    return shape.to(dtype=np.int32).cpu().numpy().tobytes()

# Generates a range, eg. [0, 1 ... n-1], or [start, start+1 ... end-1],
# or [start, start+step ... {highest start+step*k before end}]
def arange(*args):
    return np.arange(*args, dtype=np.int64, device=device)

# Combine two arrays along the top dimension (ie. treat them as lists
# and concatenate the lists)
def append(*args):
    if len(args[0].shape) == 1:
        return np.hstack(args)
    else:
        return np.vstack(args)

def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

# Pads an array to a given length
def pad_to(arr, new_len):
    padding = zeros((new_len - arr.shape[0],) + arr.shape[1:])
    return append(arr, padding)

# Confirms that the coefficients respect a given degree bound
def confirm_max_degree(coeffs, bound):
    return not np.any(coeffs[bound:])

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

# Convert an extension field element to a point in the extension
# field. Guarantees no properties about the point; the use case
# is purely "pick a random point" for Fiat-Shamir-style protocols
def projective_to_point(t):
    t2 = mul_ext(t, t)
    inv_1pt2 = modinv_ext((one + t2) % M31)
    return np.stack((
        mul_ext((one + M31 - t2) % M31, inv_1pt2),
        mul_ext((2 * t) % M31, inv_1pt2)
    ))

# Takes as input a N*k array of values, and a vector k extension field
# elements, and computes the size-N linear combination
def fold(vector, fold_factors):
    return np.sum(
        vector.reshape(vector.shape+(1,)) * fold_factors % M31,
        axis=-2
    ) % M31

# Same put takes as input a N*k array of extension field elements
def fold_ext(vector, fold_factors):
    return np.sum(mul_ext(vector, fold_factors), axis=-2) % M31

# Multiplication, mod 2**31-1
def mul(x, y):
    return x*y % M31

# Multiplication in the extension field
def mul_ext(A, B):
    # Karatsuba does not seem to actually speed this up!!
    A = A.swapaxes(0, A.ndim-1)
    B = B.swapaxes(0, B.ndim-1)
    o_LL_r = (A[0] * B[0] - A[1] * B[1]) % M31
    o_LL_i = (A[0] * B[1] + A[1] * B[0]) % M31
    o_LR_r = A[0] * B[2] - A[1] * B[3]
    o_LR_i = A[0] * B[3] + A[1] * B[2] - M31SQ
    o_RL_r = A[2] * B[0] - A[3] * B[1]
    o_RL_i = A[2] * B[1] + A[3] * B[0] - M31SQ
    o_RR_r = (A[2] * B[2] - A[3] * B[3]) % M31
    o_RR_i = (A[2] * B[3] + A[3] * B[2]) % M31
    o = np.stack((
        o_LL_r - o_RR_r + o_RR_i * 2,
        o_LL_i - o_RR_i - o_RR_r * 2,
        o_LR_r + o_RL_r,
        o_LR_i + o_RL_i
    ))
    return (o % M31).swapaxes(0, o.ndim-1)

# Modular inverse, mod 2**31-1
def modinv(x):
    o = x
    pow_of_x = (x * x) % M31
    for i in range(29):
        pow_of_x = (pow_of_x * pow_of_x) % M31
        o = (o * pow_of_x) % M31
    return o

# Modular inverse in the extension field
# Uses the formula 1/(a+bi) = (a-bi)/(a^2+b^2),
# together with an analog of that formula for the second
# extension (add w where w^2 = -1-2i)
def modinv_ext(x):
    x = x.swapaxes(0, x.ndim-1)
    r20 = (x[2] * x[2] + M31SQ - x[3] * x[3]) % M31
    r21 = (2 * x[2] * x[3]) % M31
    denom0 = (x[0]**2 - x[1]**2 + r20 - r21 * 2) % M31
    denom1 = (2*x[0]*x[1] + r21 + r20 * 2) % M31
    inv_denom_norm = modinv((denom0 ** 2 + denom1 ** 2) % M31)
    inv_denom0 = (denom0 * inv_denom_norm) % M31
    inv_denom1 = (M31SQ - denom1 * inv_denom_norm) % M31
    o = np.stack((
        x[0] * inv_denom0 + M31SQ - x[1] * inv_denom1,
        x[0] * inv_denom1 + x[1] * inv_denom0,
        M31SQ - x[2] * inv_denom0 + x[3] * inv_denom1,
        M31SQ * 2 - x[2] * inv_denom1 - x[3] * inv_denom0,
    ))
    return (o % M31).swapaxes(0, o.ndim-1)

one = array([1,0,0,0])
m31_arith = (
    array(1),
    lambda *x: sum(x) % M31,
    mul
)
ext_arith = (
    one,
    lambda *x: sum(x) % M31,
    mul_ext
)

# Evaluates the simplest polynomial that equals zero across
# the domain of size `degree`, at the given coords. Supports
# the base field or extension field
def eval_zpoly_at(degree, coords, arith):
    one, add, mul = arith
    Z = coords[...,0,:] if one.ndim == 1 else coords[...,0]
    depth = Z.ndim - one.ndim
    for i in range(1, log2(degree)):
        Z = (2 * mul(Z, Z) - one.reshape((1,)*depth + one.shape)) % M31
    return Z

# Adds two points on the circle (this is angular addition, similar to
# trigonometry sin(x+y) = sinxcosy + cosxsiny, cos(x+y) = cosxcosy - sinxsiny
def point_add(pt1, pt2):
    o = zeros(max(pt1.shape, pt2.shape))
    o[...,0] = (M31SQ + pt1[...,0]*pt2[...,0] - pt1[...,1]*pt2[...,1]) % M31
    o[...,1] = (pt1[...,0]*pt2[...,1] + pt1[...,1]*pt2[...,0]) % M31
    return o

# Adds a point on the circle to itself, similar to trigonometry
# sin(2x) = 2sinxcosx, cos(2x) = 2cos^2x-1
def point_double(pt):
    o = np.zeros_like(pt)
    o[...,0] = (2 * pt[...,0] * pt[...,0] - 1) % M31
    o[...,1] = (2 * pt[...,0] * pt[...,1]) % M31
    return o

# Add two points whose coordinates are in the extension field
def point_add_ext(pt1, pt2):
    return np.stack((
        mul_ext(pt1[0], pt2[0])
        + M31SQ - mul_ext(pt1[1], pt2[1]),
        mul_ext(pt1[0], pt2[1]) +
        mul_ext(pt1[1], pt2[0])
    )) % M31

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
        hash(entropy + bytes([i])) for i in range((num_challenges + 7) // 8)
    )
    return array([
        int.from_bytes(challenge_data[i:i+4], 'little') % domain_size
        for i in range(0, num_challenges * 4, 4)
    ])

# Generates some pseudorandom numbers mod 2**31
def mk_junk_data(length):
    a = arange(length, length*2)
    return ((3**a) ^ (7**a)) % M31
