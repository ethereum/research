import torch as np
np.array = np.tensor
np.array_equal = np.equal
np.copy = np.clone
device = np.device("cuda" if np.cuda.is_available() else "cpu")
#try:
#    import cupy as np
#except:
#    import numpy as np

M31 = 2**31-1
M31SQ = (2**31-1)**2
TOP_DOMAIN_SIZE = 2**24
HALF = 2**30
EXTENSION_I = 2

def array(x):
    return np.array(x, dtype=np.int64, device=device)

def zeros(shape):
    return np.zeros(shape, dtype=np.int64, device=device)

def tobytes(shape):
    return shape.to(dtype=np.int32).cpu().numpy().tobytes()

def arange(*args):
    return np.arange(*args, dtype=np.int64, device=device)

def append(*args):
    if len(args[0].shape) == 1:
        return np.hstack(args)
    else:
        return np.vstack(args)

def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

def point_add(pt1, pt2):
    o = zeros(max(pt1.shape, pt2.shape))
    o[...,0] = (M31SQ + pt1[...,0]*pt2[...,0] - pt1[...,1]*pt2[...,1]) % M31
    o[...,1] = (pt1[...,0]*pt2[...,1] + pt1[...,1]*pt2[...,0]) % M31
    return o

def point_double(pt):
    o = np.zeros_like(pt)
    o[...,0] = (2 * pt[...,0] * pt[...,0] - 1) % M31
    o[...,1] = (2 * pt[...,0] * pt[...,1]) % M31
    return o

G = array([1268011823, 2])
for i in range(log2(TOP_DOMAIN_SIZE), log2(M31+1)-1):
    G = point_double(G)

top_domain = zeros((2,) + G.shape)
top_domain[0][0] = 1
top_domain[1] = G
for i in range(1, log2(TOP_DOMAIN_SIZE * 2)):
    new_domain = zeros((2**(i+1),) + G.shape)
    new_domain[::2] = point_double(top_domain)
    new_domain[1::2] = point_add(G, new_domain[::2])
    top_domain = new_domain
top_domain = top_domain[1::2]
#top_domain = point_multiply(G, np.arange(1, TOP_DOMAIN_SIZE*2, 2))
sub_domains = zeros((TOP_DOMAIN_SIZE*2, 2))
sub_domains[TOP_DOMAIN_SIZE:] = top_domain
for i in range(log2(TOP_DOMAIN_SIZE)-1, -1, -1):
    sub_domains[2**i:2**(i+1)] = point_double(sub_domains[2**(i+1):(2**i)*3])

def modinv(x):
    o = x
    pow_of_x = (x * x) % M31
    for i in range(29):
        pow_of_x = (pow_of_x * pow_of_x) % M31
        o = (o * pow_of_x) % M31
    return o

xfac = sub_domains[:,0]
yfac = sub_domains[:,1]
invx = modinv(xfac)
invy = modinv(yfac)

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

rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    rbos[2**i:2**(i+1)] = reverse_bit_order(arange(2**i))

folded_rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    folded_rbos[2**i:2**(i+1)] = folded_reverse_bit_order(arange(2**i))

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
        twiddle_box = np.zeros_like(L)
        twiddle_box[:] = twiddle.reshape((1, half_len) + (1,) * (L.ndim - 2))
        f1 = ((L + M31 - R) * twiddle_box) % M31
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    inv_size = (1 << (31-log2(size))) % M31
    return (
        (vals.reshape((size,) + shape_suffix))[rbos[size:size*2]]
        * inv_size
     ) % M31

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
            twiddle = yfac[full_len: full_len + half_len]
        else:
            twiddle = xfac[full_len*2: full_len*2 + half_len]
        twiddle_box = np.zeros_like(f0)
        twiddle_box[:] = twiddle.reshape((1, half_len) + (1,) * (f0.ndim - 2))
        L = (f0 + f1 * twiddle_box) % M31
        R = (f0 + M31SQ - f1 * twiddle_box) % M31
        vals[:, :half_len] = L
        vals[:, half_len:] = np.flip(R, (1,))
    return np.reshape(vals, (size,) + shape_suffix)

def to_extension_field(values):
    o = zeros(values.shape + (4,))
    v = values.reshape(values.shape+(1,))
    o[...,:1] = v
    return o

def karat2(a, b):
    z1 = a[0] * b[0]
    z2 = a[1] * b[1]
    z3 = (a[0] + a[1]) * (b[0] + b[1])
    return [(z1 + M31SQ - z2) % M31, (z3 - z1 - z2) % M31]

def extension_field_mul(A, B):
    # todo: needs moar karatsuba
    A = A.swapaxes(0, A.ndim-1)
    B = B.swapaxes(0, B.ndim-1)
    #o_LL = karat2(A[0], A[1], B[0], B[1])
    #o_LR = karat2(A[0], A[1], B[2], B[3])
    #o_RL = karat2(A[2], A[3], B[0], B[1])
    #o_z3 = karat2((A[0] + A[2]) % M31, (A[1] + A[3]) % M31, (B[0] + B[2]) % M31, (B[1] + B[3]) % M31)
    #o_RR = karat2(A[2:], B[2:])
    #o = np.array([
    #    o_LL[0] + M31SQ - (o_RR[0] % M31) + (o_RR[1] % M31) * EXTENSION_I,
    #    o_LL[1] + M31SQ - (o_RR[1] % M31) - (o_RR[0] % M31) * EXTENSION_I,
    #    (o_z3[0] + M31SQ) - (o_LL[0] + o_RR[0]) % M31,
    #    (o_z3[1] + M31SQ) - (o_LL[1] + o_RR[1]) % M31,
    #], dtype=np.int64)
    #z1 = A[::2] * B[::2]
    #z2 = A[1::2] * B[1::2]
    #z3 = (A[::2] + A[1::2]) * (B[::2] + B[1::2])
    #oR = (z1[np.arange(2), np.arange(2)] + M31SQ - z2[np.arange(2), np.arange(2)]) % M31
    #oI = (z3[np.arange(2), np.arange(2)] - z1[np.arange(2), np.arange(2)] - z2[np.arange(2), np.arange(2)]) % M31
    #o = np.array([
    #    oR[0,0] + M31SQ - oR[1,1] + oI[1,1] * 2,
    #    oI[0,0] + M31SQ - oI[1,1] - oR[1,1] * 2,
    #    oR[0,1] * oR[1,0],
    #    oI[0,1] * oI[1,0]
    #])
    o_LL_r = (A[0] * B[0] - A[1] * B[1]) % M31
    o_LL_i = (A[0] * B[1] + A[1] * B[0]) % M31
    o_LR_r = A[0] * B[2] - A[1] * B[3]
    o_LR_i = A[0] * B[3] + A[1] * B[2] - M31SQ
    o_RL_r = A[2] * B[0] - A[3] * B[1]
    o_RL_i = A[2] * B[1] + A[3] * B[0] - M31SQ
    o_RR_r = (A[2] * B[2] - A[3] * B[3]) % M31
    o_RR_i = (A[2] * B[3] + A[3] * B[2]) % M31
    o = np.stack((
        o_LL_r - o_RR_r + o_RR_i * EXTENSION_I,
        o_LL_i - o_RR_i - o_RR_r * EXTENSION_I,
        o_LR_r + o_RL_r,
        o_LR_i + o_RL_i
        #(o_z3_r + M31SQ) - (o_LL_r + o_RR_r) % M31,
        #(o_z3_i + M31SQ) - (o_LL_i + o_RR_i) % M31,
    ))
    return (o % M31).swapaxes(0, o.ndim-1)

one = array([1,0,0,0])

def to_ext_if_needed(obj, object_dim):
    if obj.ndim == object_dim:
        return to_extension_field(obj)
    else:
        return obj

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
        twiddle_box = np.zeros_like(L)
        twiddle_box[:] = twiddle.reshape((half_len,) + (1,) * (L.ndim - 1))
        f1 = ((L + M31 - R) * twiddle_box) % M31
        vals = (f0 + mul(baryfac, f1)) % M31
    inv_size = (1 << (31-log2(size))) % M31
    return (vals[0] * inv_size) % M31
