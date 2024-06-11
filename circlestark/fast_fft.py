try:
    import cupy as np
except:
    import numpy as np

M31 = 2**31-1
M31SQ = (2**31-1)**2
TOP_DOMAIN_SIZE = 2**21
HALF = 2**30
EXTENSION_I = 2

def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

def point_add(pt1, pt2):
    o = np.zeros(max(pt1.shape, pt2.shape), dtype=np.uint64)
    o[...,0] = (M31SQ + pt1[...,0]*pt2[...,0] - pt1[...,1]*pt2[...,1]) % M31
    o[...,1] = (M31SQ + pt1[...,0]*pt2[...,1] + pt1[...,1]*pt2[...,0]) % M31
    return o

def point_double(pt):
    o = np.zeros_like(pt)
    o[...,0] = (M31 + 2 * pt[...,0] * pt[...,0] - 1) % M31
    o[...,1] = (2 * pt[...,0] * pt[...,1]) % M31
    return o

G = np.array([1268011823, 2], dtype=np.uint64)
for i in range(log2(TOP_DOMAIN_SIZE), log2(M31+1)-1):
    G = point_double(G)

def point_multiply(pt, n):
    n = np.array(n, dtype=np.uint64)
    if n.shape < pt.shape[:-1]:
        n = np.broadcast_to(n, pt.shape[:-1])
    n = np.repeat(np.expand_dims(n, axis=-1), 2, axis=-1)
    zeros = np.zeros_like(pt)
    zeros[...,0] = 1
    o = np.zeros_like(pt)
    o[...,0] = 1
    rounds = int(np.max(n)).bit_length()
    for i in range(rounds):
        #print("Round {}".format(i))
        o = point_add(o,
            np.broadcast_to(pt, n.shape) * ((n >> i) % 2) + 
            np.broadcast_to(zeros, n.shape) * (1 - ((n >> i) % 2))
        )
        pt = point_double(pt)
    return o

top_domain = point_multiply(G, np.arange(1, TOP_DOMAIN_SIZE*2, 2))
sub_domains = np.zeros((TOP_DOMAIN_SIZE*2, 2), dtype=np.uint64)
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
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    return (
        vals.reshape((2,)*log2(size) + shape_suffix)
            .transpose(
                tuple(range(log2(size)-1,-1,-1))
                + tuple(range(log2(size), log2(size) + len(shape_suffix)))
            ).reshape((size,) + shape_suffix)
    )

def fft(vals, is_top_level=True):
    vals = np.array(vals, dtype=np.uint64)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        L = vals[:, :half_len]
        R = vals[:, full_len-1:half_len-1:-1]
        f0 = ((L + R) * HALF) % M31
        if i==0 and is_top_level:
            twiddle = invy[full_len: full_len + half_len]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
        twiddle = np.expand_dims(
            np.broadcast_to(twiddle, (1 << i, L.shape[1])),
            axis=tuple(range(2, len(L.shape)))
        )
        f1 = ((((L + M31 - R) * HALF) % M31) * twiddle) % M31
        vals[:, :half_len] = f0
        vals[:, half_len:] = f1
    return reverse_bit_order(vals.reshape((size,) + shape_suffix))

def bary_eval(vals, pt):
    vals = np.array(vals, dtype=np.uint64)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = vals[full_len-1:half_len-1:-1]
        f0 = ((L + R) * HALF) % M31
        if i==0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt[1]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            baryfac = pt[0]
            for _ in range(i-1):
                baryfac = (2 * baryfac**2 + M31 - 1) % M31
        twiddle = np.expand_dims(
            np.broadcast_to(twiddle, (L.shape[0],)),
            axis=tuple(range(1, len(L.shape)))
        )
        f1 = ((((L + M31 - R) * HALF) % M31) * twiddle) % M31
        vals = (f0 + baryfac * f1) % M31
    return vals[0]

def inv_fft(vals):
    vals = np.array(vals, dtype=np.uint64)
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
        twiddle = np.expand_dims(
            np.broadcast_to(twiddle, (1 << i, f0.shape[1])),
            axis=tuple(range(2, len(f0.shape)))
        )
        L = (f0 + f1 * twiddle) % M31
        R = (f0 + M31SQ - f1 * twiddle) % M31
        vals[:, :half_len] = L
        vals[:, full_len-1:half_len-1:-1] = R
    return np.reshape(vals, (size,) + shape_suffix)

def to_extension_field(values):
    return np.pad(values[...,np.newaxis], ((0,0), (0,3)))

def extension_field_mul(A, B):
    # todo: needs moar karatsuba
    A = A.transpose()
    B = B.transpose()
    o_LL = [A[0] * B[0] + M31SQ - A[1] * B[1], A[0] * B[1] + A[1] * B[0]]
    o_LR = [A[0] * B[2] + M31SQ - A[1] * B[3], A[0] * B[3] + A[1] * B[2]]
    o_RL = [A[2] * B[0] + M31SQ - A[3] * B[1], A[2] * B[1] + A[3] * B[0]]
    o_RR = [A[2] * B[2] + M31SQ - A[3] * B[3], A[2] * B[3] + A[3] * B[2]]
    o = np.array([
        o_LL[0] + M31SQ - (o_RR[0] % M31) + (o_RR[1] % M31) * EXTENSION_I,
        o_LL[1] + M31SQ - (o_RR[1] % M31) - (o_RR[0] % M31) * EXTENSION_I,
        o_LR[0] + o_RL[0],
        o_LR[1] + o_RL[1]
    ], dtype=np.uint64)
    return (o % M31).transpose()

def bary_eval(vals, pt):
    vals = np.array(vals, dtype=np.uint64)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = vals[full_len-1:half_len-1:-1]
        f0 = ((L + R) * HALF) % M31
        if i==0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt[1]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            baryfac = pt[0]
            for _ in range(i-1):
                baryfac = (2 * baryfac**2 + M31 - 1) % M31
        twiddle = np.expand_dims(
            np.broadcast_to(twiddle, (L.shape[0],)),
            axis=tuple(range(1, len(L.shape)))
        )
        f1 = ((((L + M31 - R) * HALF) % M31) * twiddle) % M31
        vals = (f0 + baryfac * f1) % M31
    return vals[0]

def bary_eval_ext(vals, pt):
    vals = np.array(vals, dtype=np.uint64)
    shape_suffix = vals.shape[1:]
    size = vals.shape[0]
    for i in range(log2(size)):
        #vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[0]
        half_len = full_len >> 1
        L = vals[:half_len]
        R = vals[full_len-1:half_len-1:-1]
        f0 = ((L + R) * HALF) % M31
        if i==0:
            twiddle = invy[full_len: full_len + half_len]
            baryfac = pt[1]
        else:
            twiddle = invx[full_len*2: full_len*2 + half_len]
            baryfac = pt[0]
            for _ in range(i-1):
                baryfac = (
                    (2 * extension_field_mul(baryfac, baryfac) + M31 - 1)
                    % M31
                )
        twiddle = np.expand_dims(
            np.broadcast_to(twiddle, (L.shape[0],)),
            axis=tuple(range(1, len(L.shape)))
        )
        f1 = ((((L + M31 - R) * HALF) % M31) * twiddle) % M31
        vals = (f0 + extension_field_mul(baryfac, f1)) % M31
    return vals[0]
