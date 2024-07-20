from utils import (
    np, array, zeros, tobytes, arange, append, log2,
    reverse_bit_order,
    to_ext_if_needed, to_extension_field
)
from zorch.m31 import (
    zeros, array, arange, append, tobytes, add, sub, mul, cp as np,
    mul_ext, modinv_ext, sum as m31_sum, eq, iszero, M31
)
from zorch import m31
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
        f0 = m31.add(L, R)
        if i==0 and is_top_level:
            twiddle = invy[full_len: full_len + half_len]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
        twiddle_box = twiddle.reshape((1, half_len) + (1,) * (L.ndim - 2))
        f1 = m31.mul(m31.sub(L, R), twiddle_box)
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    inv_size = np.array((1 << (31-log2(size))) % m31.M31, dtype=np.uint32)
    return m31.mul(
        (vals.reshape((size,) + shape_suffix))[rbos[size:size*2]],
        inv_size
    )

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
            twiddle = sub_domains[full_len: full_len + half_len].y
        else:
            twiddle = sub_domains[full_len*2: full_len*2 + half_len].x
        f1_times_twiddle = m31.mul(
            f1,
            twiddle.reshape((1, half_len) + (1,) * (f0.ndim - 2))
        )
        L = m31.add(f0, f1_times_twiddle)
        R = m31.sub(f0, f1_times_twiddle)
        vals[:, :half_len] = L
        vals[:, half_len:] = np.flip(R, (1,))
    return np.reshape(vals, (size,) + shape_suffix)

# Given a list of evaluations, computes the evaluation of that polynomial at
# one point. The point can be in the base field or extension field
def bary_eval(vals, pt, is_extended=False, first_round_optimize=False):
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    if is_extended:
        pt = pt.to_extended()
        mul = m31.mul_ext
        one = array([1,0,0,0])
    else:
        mul = m31.mul
        one = array(1)
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = np.flip(vals[half_len:], (0,))
        f0 = m31.add(L, R)
        if i == 0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt.y
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            if i == 1:
                baryfac = pt.x
            else:
                baryfac = m31.sub(2 * mul(baryfac, baryfac) % M31, one)
        twiddle_box = twiddle.reshape((half_len,) + (1,) * (L.ndim - 1))
        f1 = m31.mul(m31.sub(L, R), twiddle_box)
        if first_round_optimize and i==0 and one.ndim==1:
            vals = m31.add(
                to_extension_field(f0),
                m31.mul(baryfac, f1.reshape(f1.shape+(1,)))
            )
        else:
            vals = m31.add(f0, mul(baryfac, f1))
    inv_size = (1 << (31-log2(size))) % M31
    return m31.mul(vals[0], np.array(inv_size, dtype=np.uint32))
