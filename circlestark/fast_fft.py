from utils import (
    np, array, zeros, tobytes, arange, append, log2, point_add,
    point_double, modinv, one, M31, reverse_bit_order,
    to_ext_if_needed
)
from precomputes import rbos, invx, invy, sub_domains

# Converts a list of evaluations to a list of coefficients. Note that the
# coefficients are in a "weird" basis: 1, y, x, xy, 2x^2-1...
def fft(vals, is_top_level=True):
    vals = np.copy(vals)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, :half_len]
        R = np.flip(vals[:, half_len:], (1,))
        f0 = (L + R) % M31
        if i==0 and is_top_level:
            twiddle = invy[full_len: full_len + half_len]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
        twiddle_box = twiddle.reshape((1, half_len) + (1,) * (L.ndim - 2))
        f1 = ((L + M31 - R) * twiddle_box) % M31
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    inv_size = (1 << (31-log2(size))) % M31
    return (
        (vals.reshape((size,) + shape_suffix))[rbos[size:size*2]]
        * inv_size
     ) % M31

# Converts a list of coefficients into a list of evaluations
def inv_fft(vals):
    vals = np.copy(vals)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    vals = reverse_bit_order(vals)
    for i in range(log2(size)-1, -1, -1):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        f0 = vals[:, :half_len]
        f1 = vals[:, half_len:]
        if i==0:
            twiddle = sub_domains[full_len: full_len + half_len, 1]
        else:
            twiddle = sub_domains[full_len*2: full_len*2 + half_len, 0]
        twiddle_box = twiddle.reshape((1, half_len) + (1,) * (f0.ndim - 2))
        L = (f0 + f1 * twiddle_box) % M31
        R = (f0 - f1 * twiddle_box) % M31
        vals[:, :half_len] = L
        vals[:, half_len:] = np.flip(R, (1,))
    return np.reshape(vals, (size,) + shape_suffix)

# Given a list of evaluations, computes the evaluation of that polynomial at
# one point. The point can be in the base field or extension field
def bary_eval(vals, pt, arith):
    one, add, mul = arith
    vals = np.copy(vals)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    if one.ndim == 1:
        pt = to_ext_if_needed(pt, object_dim=1)
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = np.flip(vals[half_len:], (0,))
        f0 = (L + R) % M31
        if i == 0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt[1]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            if i == 1:
                baryfac = pt[0]
            else:
                baryfac = (2 * mul(baryfac, baryfac) - one) % M31
        twiddle_box = twiddle.reshape((half_len,) + (1,) * (L.ndim - 1))
        f1 = ((L + M31 - R) * twiddle_box) % M31
        vals = (f0 + mul(baryfac, f1)) % M31
    inv_size = (1 << (31-log2(size))) % M31
    return (vals[0] * inv_size) % M31
