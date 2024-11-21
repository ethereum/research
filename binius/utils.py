# Helper methods to allow the below methods to be used for multiple types
# (binary fields, prime fields, regular integers), while still enforcing
# compatibility
#
# The type flexibility also lets us reuse much of the same code for
# packed-binius that we use for simple-binius

def get_class(arg, start=int):
    if isinstance(arg, (list, tuple)):
        output = start
        for a in arg:
            output = get_class(a, output)
        return output
    elif start == int:
        return arg.__class__
    elif arg.__class__ == int:
        return start
    elif start == arg.__class__:
        return arg.__class__
    else:
        raise Exception("Incompatible classes: {} {}".format(start, arg.__class__))

def spread_type(arg, cls):
    if isinstance(arg, cls):
        return arg
    elif isinstance(arg, int):
        return cls(arg)
    elif isinstance(arg, (list, tuple)):
        return arg.__class__([spread_type(item, cls) for item in arg])
    else:
        raise Exception("Type propagation of {} hit incompatible element: {}".format(cls, arg))

def enforce_type_compatibility(*args):
    cls = get_class(args)
    return tuple([cls] + list(spread_type(arg, cls) for arg in args))

# Evaluate a (univariate) polynomial at the given point
# eg (over regular integers):
#
# >>> u.eval_poly_at([3, 1, 4, 1, 5], 10)
# 51413

def eval_poly_at(poly, pt):
    cls, poly, pt = enforce_type_compatibility(poly, pt)
    o = cls(0)
    power = cls(1)
    for coeff in poly:
        o += coeff * power
        power *= pt
    return o

def add_polys(a, b):
    cls, a, b = enforce_type_compatibility(a, b)
    minl, maxl = sorted((len(a), len(b)))
    tail = [cls(0)] * (maxl - minl)
    a += tail    
    b += tail    
    return [(a[i] + b[i]) for i in range(maxl)]

# Multiply two polynomials together
# eg (over regular integers):
#
# >>> u.mul_polys([-3, 1], [3, 1])
# [-9, 0, 1]

def mul_polys(a, b):
    cls, a, b = enforce_type_compatibility(a, b)
    o = [cls(0)] * (len(a) + len(b) - 1)
    for i, aval in enumerate(a):
        if aval != 0:
            for j, bval in enumerate(b):
                o[i+j] += aval * bval
    return o

# Computes the polynomial the equals 0 over 0...size-1, except at pt,
# where it equals 1.
# eg (over regular integers):
#
# >>> u.compute_lagrange_poly(4, 0)
# [1.0, -1.8333333333333333, 1.0, -0.16666666666666666]
#
# Plugging x={0,1,2,3,4,5} into this polynomial gives y={1,0,0,0,-1,-4}
# (approximately, since floats are inexact), as expected

def compute_lagrange_poly(size, pt):
    cls = get_class(pt)
    opoly = [cls(1)]
    ofactor = cls(1)
    for i in range(size):
        _i = cls(i)
        if _i != pt:
            opoly = mul_polys(opoly, [-_i, 1])
            ofactor *= (pt - _i)
    return [x/ofactor for x in opoly]

# Treat `evals` as the evaluations of a multilinear polynomial over {0,1}^k.
# That is, if evals is [a,b,c,d], then a=P(0,0), b=P(1,0), c=P(0,1), d=P(1,1)
# Evaluate that polynomial at pt
#
# Example (over regular integers):
#
# >>> u.multilinear_poly_eval([3, 14, 15, 92], [0,0])
# 3
# >>> u.multilinear_poly_eval([3, 14, 15, 92], [1,0])
# 14
# >>> u.multilinear_poly_eval([3, 14, 15, 92], [2, 5])
# 745

def multilinear_poly_eval(evals, pt):
    cls, evals, pt = enforce_type_compatibility(evals, pt)
    assert len(evals) == 2 ** len(pt)
    return _multilinear_poly_eval(cls, evals, pt)

def _multilinear_poly_eval(cls, evals, pt):
    if len(pt) == 0:
        return evals[0]
    top = _multilinear_poly_eval(cls, evals[:len(evals)//2], pt[:-1])
    bottom = _multilinear_poly_eval(cls, evals[len(evals)//2:], pt[:-1])
    return (
        (bottom - top) * pt[-1] + top
    )

# Uses a Reed-Solomon code to extend the input list of N values into a list of
# 2N values. That is, treat the input as P(0) ... P(N-1) for some polynomial
# P, and then append P(N) ... P(2N-1).
#
# Example (over regular integers):
#
# >>> u.extend([1, 4, 9, 16])
# [1, 4, 9, 16, 24.999999999999986, 35.99999999999997, 49.0, 64.0]

def extend(vals, expansion_factor=2):
    cls, vals = enforce_type_compatibility(vals)
    lagranges = [
        compute_lagrange_poly(len(vals), cls(i))
        for i in range(len(vals))
    ]
    output = vals[::]
    for x in range(len(vals), expansion_factor * len(vals)):
        o = cls(0)
        for v, L in zip(vals, lagranges):
            o += v * eval_poly_at(L, x)
        output.append(o)
    return output

# Returns the 2^k-long list of all possible results of walking through pt
# (an evaluation point) and at each step taking either coord or 1-coord.
# This is a natural companion method to `multilinear_poly_eval`, because
# it gives a list where `output[i]` equals
# `multilinear_poly_eval([0, 0 ... 1 ... 0, 0], pt)`, where the 1 is in
# position i.
#
# Example (over regular integers):
#
# >>> u.evaluation_tensor_product([2, 5])
# [4, -8, -5, 10]
# >>> u.multilinear_poly_eval([1,0,0,0], [2,5])
# 4
# >>> u.multilinear_poly_eval([0,1,0,0], [2,5])
# -8

def evaluation_tensor_product(pt):
    cls, pt = enforce_type_compatibility(pt)
    o = [cls(1)]
    for coord in pt:
        o = [
            (cls(1) - coord) * v for v in o
        ] + [
            coord * v for v in o
        ]
    return o

# 1 -> 0
# 2 -> 1
# 64 -> 6
# 175 -> error (only takes powers of 2)
def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

# Packs a list of bits into a smaller list of binary representations
def pack_vector(bits, slice_size):
    o = []
    for pos in range(0, len(bits), slice_size):
        new_value = 0
        for v in reversed(bits[pos:pos+slice_size]):
            assert v == 0 or v == 1
            new_value = (new_value * 2) + (1 if v == 1 else 0)
        o.append(new_value)
    _, _, o = enforce_type_compatibility(bits, o)
    return o
