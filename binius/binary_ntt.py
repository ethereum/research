from binary_fields import BinaryFieldElement as B
from utils import log2, eval_poly_at, mul_polys, add_polys

# Computes the polynomial that returns 0 on 0.....(1<<i)-1 and 1 on (1<<i)
# Relies on the identity W{i+1}(X) = Wi(X) * (Wi(X) + Wi(1<<i))
def get_Wi(i):
    if i == 0:
        return [B(0), B(1)]
    else:
        prev = get_Wi(i - 1)
        o = mul_polys(prev, add_polys(prev, [B(1)]))
        inv_quot = eval_poly_at(o, B(1<<i)).inv()
        return [x*inv_quot for x in o]

# Maintains a cache of Wi(pt) values, so Wi_eval_cache[dim][pt] = W{dim}(pt)
Wi_eval_cache = []

def get_Wi_eval(dim, pt):
    coord = B(pt).value
    while len(Wi_eval_cache) <= dim:
        Wi_eval_cache.append({})
    if dim == 0:
        return B(pt)
    # The same method as above, but applied directly to evaluate at `pt`
    if coord not in Wi_eval_cache[dim]:
        prev = get_Wi_eval(dim-1, pt)
        prev_quot = get_Wi_eval(dim-1, 1<<dim)
        Wi_eval_cache[dim][coord] = (
            prev * (prev + B(1)) /
            (prev_quot * (prev_quot + B(1)))
        )
    return Wi_eval_cache[dim][coord]

# We define a "basis", where B_{2**k} = W_k, and any other B_n is the product
# of the power-of-two B_{k1} * ... * B_{kn} where k1...kn are powers of two
# that sum to n
def get_Bi(i):
    opoly = [B(1)]
    for j, bit in enumerate(bin(i)[:1:-1]):
        if bit == '1':
            opoly = mul_polys(opoly, get_Wi(j))
    return opoly

# Gets all B_i values up to (but not including) `bits`
def get_basis(bits):
    return [get_Bi(i) for i in range(bits)]

# Treat the input as coefficients of a polynomial, with each coefficient to be
# multiplied by Bk(i), as opposed to i**k
def eval_poly_in_basis(poly, i):
    basis = get_basis(len(poly))
    o = B(0)
    for coeff, basisitem in zip(poly, basis):
        o += eval_poly_at(basisitem, i) * coeff
    return o

# Converts a polynomial with coefficients in the above basis, into evaluations 
# See page 4 of https://arxiv.org/pdf/1802.03932

def additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L, R = vals[:halflen], vals[halflen:]
    coeff1 = get_Wi_eval(log2(halflen), start)
    sub_input1 = [i+j*coeff1 for i,j in zip(L, R)]
    sub_input2 = [i+j for i,j in zip(sub_input1, R)]
    o = (
        additive_ntt(sub_input1, start) +
        additive_ntt(sub_input2, start + halflen)
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

# Converts evaluations into coefficients (in the above basis) of a polynomial
def inv_additive_ntt(vals, start=0):
    vals = [B(val) for val in vals]
    if len(vals) == 1:
        return vals
    halflen = len(vals)//2
    L = inv_additive_ntt(vals[:halflen], start)
    R = inv_additive_ntt(vals[halflen:], start + halflen)
    coeff1 = get_Wi_eval(log2(halflen), start)
    coeff2 = coeff1 + 1
    o = (
        [i*coeff2+j*coeff1 for i,j in zip(L, R)] +
        [i+j for i,j in zip(L, R)]
    )
    # print('for {} at {} used coeffs {}, {}; returning {}'.format(vals, start, coeff1, coeff2, o))
    return o

# Reed-Solomon extension, using the efficient algorithms above
def extend(data, expansion_factor=2):
    data = [B(val) for val in data]
    return additive_ntt(
        inv_additive_ntt(data) +
        [B(0)] * len(data) * (expansion_factor - 1)
    )
