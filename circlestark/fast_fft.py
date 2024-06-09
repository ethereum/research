try:
    import cupy as np
except:
    import numpy as np

M31 = 2**31-1
M31SQ = (2**31-1)**2
TOP_DOMAIN_SIZE = 2**21
HALF = 2**30

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
