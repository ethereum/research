from utils import (
    cp, reverse_bit_order, log2
)

from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like, Z, G
)
from precomputes import rbos, invx, invy, sub_domains

# Converts a list of evaluations to a list of coefficients. Note that the
# coefficients are in a "weird" basis: 1, y, x, xy, 2x^2-1...
def fft(vals, is_top_level=True):
    vals = vals.copy()
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        vals = vals.reshape((1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, :half_len]
        R = vals[:, half_len:][:, ::-1, ...] # flip along axis 1
        f0 = L + R
        if i==0 and is_top_level:
            twiddle = invy[full_len: full_len + half_len]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
        twiddle_box = twiddle.reshape((1, half_len) + (1,) * (L.ndim - 2))
        f1 = (L - R) * twiddle_box
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    return (
        (vals.reshape((size,) + shape_suffix))[rbos[size:size*2]] / size
    )

# Converts a list of coefficients into a list of evaluations
def inv_fft(vals):
    vals = vals.copy()
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    vals = reverse_bit_order(vals)
    for i in range(log2(size)-1, -1, -1):
        vals = vals.reshape((1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        f0 = vals[:, :half_len]
        f1 = vals[:, half_len:]
        if i==0:
            twiddle = sub_domains[full_len: full_len + half_len].y
        else:
            twiddle = sub_domains[full_len*2: full_len*2 + half_len].x
        f1_times_twiddle = (
            f1 * twiddle.reshape((1, half_len) + (1,) * (f0.ndim - 2))
        )
        L = f0 + f1_times_twiddle
        R = f0 - f1_times_twiddle
        vals[:, :half_len] = L
        vals[:, half_len:] = R[:, ::-1, ...]
    return vals.reshape((size,) + shape_suffix)

# Given a list of evaluations, computes the evaluation of that polynomial at
# one point. The point can be in the base field or extension field
def bary_eval(vals, pt):
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = vals[half_len:][::-1]
        f0 = L + R
        if i == 0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt.y
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            if i == 1:
                baryfac = pt.x
            else:
                baryfac = baryfac * baryfac * 2 - 1
        twiddle_box = twiddle.reshape((half_len,) + (1,) * (L.ndim - 1))
        f1 = (L - R) * twiddle_box
        vals = f0 + baryfac * f1
    return vals[0] / size
