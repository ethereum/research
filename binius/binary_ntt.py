from binary_fields import BinaryFieldElement as B
from utils import log2, eval_poly_at, mul_polys

def compute_vanishing_poly(size):
    opoly = [B(1)]
    for i in range(size):
        opoly = mul_polys(opoly, [i, 1])
    return opoly

def get_Wi(i):
    p = compute_vanishing_poly(2**i)
    quot = eval_poly_at(p, 2**i)
    return [x/quot for x in p]

vpoly_cache = []

def get_vpoly_eval(dim, pt):
    coord = B(pt).value
    while len(vpoly_cache) <= dim:
        vpoly_cache.append({})
    if coord not in vpoly_cache[dim]:
        o = B(1)
        denom = B(1)
        for i in range(2**dim):
            o *= (B(coord) - i)
            denom *= (B(2**dim) - i)
        vpoly_cache[dim][coord] = o / denom
    return vpoly_cache[dim][coord]

# See page 4 of https://arxiv.org/pdf/1802.03932

def additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L, R = vals[:halflen], vals[halflen:]
    coeff1 = get_vpoly_eval(log2(halflen), start)
    coeff2 = get_vpoly_eval(log2(halflen), start + halflen)
    o = (
        additive_ntt([i+j*coeff1 for i,j in zip(L, R)], start) +
        additive_ntt([i+j*coeff2 for i,j in zip(L, R)], start + halflen)
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

def inv_additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L = inv_additive_ntt(vals[:halflen], start)
    R = inv_additive_ntt(vals[halflen:], start + halflen)
    coeff1 = get_vpoly_eval(log2(halflen), start)
    coeff2 = get_vpoly_eval(log2(halflen), start + halflen)
    o = (
        [i*coeff2+j*coeff1 for i,j in zip(L, R)] +
        [i+j for i,j in zip(L, R)]
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

def get_Bi(i):
    opoly = [B(1)]
    for j, bit in enumerate(bin(i)[:1:-1]):
        if bit == '1':
            opoly = mul_polys(opoly, get_Wi(j))
    return opoly

def get_basis(bits):
    return [get_Bi(i) for i in range(bits)]

def eval_poly_in_basis(poly, i):
    basis = get_basis(len(poly))
    o = B(0)
    for coeff, basisitem in zip(poly, basis):
        o += eval_poly_at(basisitem, i) * coeff
    return o

def extend(data, expansion_factor=2):
    data = [B(val) for val in data]
    for i in range(log2(expansion_factor)):
        data = additive_ntt(inv_additive_ntt(data) + [B(0)] * len(data))
    return data
