modulus_poly = [1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 1, 0, 0, 1,
                1]
modulus_poly_as_int = sum([(v << i) for i, v in enumerate(modulus_poly)])
degree = len(modulus_poly) - 1

two_to_the_degree = 2**degree
two_to_the_degree_m1 = 2**degree - 1

def galoistpl(a):
    # 2 is not a primitive root, so we have to use 3 as our logarithm base
    if a * 2 < two_to_the_degree:
        return (a * 2) ^ a
    else:
        return (a * 2) ^ a ^ modulus_poly_as_int

# Precomputing a log table for increased speed of addition and multiplication
glogtable = [0] * (two_to_the_degree)
gexptable = []
v = 1
for i in range(two_to_the_degree_m1):
    glogtable[v] = i
    gexptable.append(v)
    v = galoistpl(v)

gexptable += gexptable + gexptable

# Add two values in the Galois field
def galois_add(x, y):
    return x ^ y

# In binary fields, addition and subtraction are the same thing
galois_sub = galois_add

# Multiply two values in the Galois field
def galois_mul(x, y):
    return 0 if x*y == 0 else gexptable[glogtable[x] + glogtable[y]]

# Divide two values in the Galois field
def galois_div(x, y):
    return 0 if x == 0 else gexptable[(glogtable[x] - glogtable[y]) % two_to_the_degree_m1]

# Evaluate a polynomial at a point
def eval_poly_at(p, x):
    if x == 0:
        return p[0]
    y = 0
    logx = glogtable[x]
    for i, p_coeff in enumerate(p):
        if p_coeff:
            # Add x**i * coeff
            y ^= gexptable[(logx * i + glogtable[p_coeff]) % two_to_the_degree_m1]
    return y


# Given p+1 y values and x values with no errors, recovers the original
# p+1 degree polynomial.
# Lagrange interpolation works roughly in the following way.
# 1. Suppose you have a set of points, eg. x = [1, 2, 3], y = [2, 5, 10]
# 2. For each x, generate a polynomial which equals its corresponding
#    y coordinate at that point and 0 at all other points provided.
# 3. Add these polynomials together.

def lagrange_interp(pieces, xs):
    # Generate master numerator polynomial, eg. (x - x1) * (x - x2) * ... * (x - xn)
    root = mk_root_2(xs)
    #print(root)
    assert len(root) == len(pieces) + 1
    # print(root)
    # Generate the derivative
    d = derivative(root)
    # Generate denominators by evaluating numerator polys at each x
    denoms = multi_eval_2(d, xs)
    print(denoms)
    # denoms = [eval_poly_at(d, xs[i]) for i in range(len(xs))]
    # Generate output polynomial, which is the sum of the per-value numerator
    # polynomials rescaled to have the right y values
    factors = [galois_div(p, d) for p, d in zip(pieces, denoms)]
    o = multi_root_derive(xs, factors)
    # print(o)
    return o

def multi_root_derive(xs, muls):
    if len(xs) == 1:
        return [muls[0]]
    R1 = mk_root_2(xs[:len(xs) // 2])
    R2 = mk_root_2(xs[len(xs) // 2:])
    x1 = karatsuba_mul(R1, multi_root_derive(xs[len(xs) // 2:], muls[len(muls) // 2:]) + [0])
    x2 = karatsuba_mul(R2, multi_root_derive(xs[:len(xs) // 2], muls[:len(muls) // 2]) + [0])
    o = [v1 ^ v2 for v1, v2 in zip(x1, x2)][:len(xs)]
    # print(len(R1), len(x1), len(xs), len(o))
    return o

def multi_root_derive_1(xs, muls):
    o = [0] * len(xs)
    for i in range(len(xs)):
        _xs = xs[:i] + xs[(i+1):]
        root = mk_root_2(_xs)
        for j in range(len(root)):
            o[j] ^= galois_mul(root[j], muls[i])
    return o

a = 124
b = 8932
c = 12415

assert galois_mul(galois_add(a, b), c) == galois_add(galois_mul(a, c), galois_mul(b, c))

def karatsuba_mul(p1, p2):
    L = len(p1)
    # assert L == len(p2)
    if L <= 16:
        o = [0] * (L * 2)
        for i, v1 in enumerate(p1):
            for j, v2 in enumerate(p2):
                if v1 and v2:
                    o[i + j] ^= gexptable[glogtable[v1] + glogtable[v2]]
        return o
    if L % 2:
        p1 = p1 + [0]
        p2 = p2 + [0]
        L += 1
    halflen = L // 2
    low1 = p1[:halflen]
    high1 = p1[halflen:]
    sum1 = [l ^ h for l, h in zip(low1, high1)]
    low2 = p2[:halflen]
    high2 = p2[halflen:]
    sum2 = [l ^ h for l, h in zip(low2, high2)]
    z2 = karatsuba_mul(high1, high2)
    z0 = karatsuba_mul(low1, low2)
    z1 = [m ^ _z0 ^ _z2 for m, _z0, _z2 in zip(karatsuba_mul(sum1, sum2), z0, z2)]
    o = z0[:halflen] + \
        [a ^ b for a, b in zip(z0[halflen:], z1[:halflen])] + \
        [a ^ b for a, b in zip(z2[:halflen], z1[halflen:])] + \
        z2[halflen:]
    return o

def mk_root_1(xs):
    root = [1]
    for x in xs:
        logx = glogtable[x]
        root.insert(0, 0)
        for j in range(len(root)-1):
            if root[j+1] and x:
                root[j] ^= gexptable[glogtable[root[j+1]] + logx]
    return root

def mk_root_2(xs):
    if len(xs) >= 128:
        return karatsuba_mul(mk_root_2(xs[:len(xs) // 2]), mk_root_2(xs[len(xs) // 2:]))[:len(xs) + 1]
    root = [1]
    for x in xs:
        logx = glogtable[x]
        root.insert(0, 0)
        for j in range(len(root)-1):
            if root[j+1] and x:
                root[j] ^= gexptable[glogtable[root[j+1]] + logx]
    return root

def derivative(root):
    return [0 if i % 2 else r for i, r in enumerate(root[1:])]

# Credit to http://people.csail.mit.edu/madhu/ST12/scribe/lect06.pdf for the algorithm
def xn_mod_poly(p):
    if len(p) == 1:
        return [galois_div(1, p[0])]
    halflen = len(p) // 2
    lowinv = xn_mod_poly(p[:halflen])
    submod_high = karatsuba_mul(lowinv, p[:halflen])[halflen:]
    med = karatsuba_mul(p[halflen:], lowinv)[:halflen]
    med_plus_high = [x ^ y for x, y in zip(med, submod_high)]
    highinv = karatsuba_mul(med_plus_high, lowinv)
    o = (lowinv + highinv)[:len(p)]
    print(halflen, lowinv, submod_high, med, highinv)
    # assert karatsuba_mul(o, p)[:len(p)] == [1] + [0] * (len(p) - 1)
    return o

def mod(a, b):
    assert len(a) == 2 * (len(b) - 1)
    L = len(b)
    inv_rev_b = xn_mod_poly(b[::-1] + [0] * (len(a) - L))[:L]
    quot = karatsuba_mul(inv_rev_b, a[::-1][:L])[:L-1][::-1]
    subt = karatsuba_mul(b, quot + [0])[:-1]
    o = [x ^ y for x, y in zip(a[:L-1], subt[:L-1])]
    # assert [x^y for x, y in zip(karatsuba_mul(quot + [0], b), o)] == a
    return o

def multi_eval_1(poly, xs):
    return [eval_poly_at(poly, x) for x in xs]

def multi_eval_2(poly, xs):
    if len(xs) <= 1024:
        return [eval_poly_at(poly, x) for x in xs]
    halflen = len(xs) // 2
    return multi_eval_2(mod(poly, mk_root_2(xs[:halflen])), xs[:halflen]) + \
           multi_eval_2(mod(poly, mk_root_2(xs[halflen:])), xs[halflen:])
           # [eval_poly_at(poly, xs[-2]), eval_poly_at(poly, xs[-1])]
