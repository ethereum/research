class Vector():
    def __init__(self, values):
        self.values = values

    def __add__(self, other):
        return Vector([v+w for v,w in zip(self.values, other.values)])

    def __mul__(self, other):
        return Vector([v*other for v in self.values])

    def __div__(self, other):
        return Vector([v/other for v in self.values])

    def __iter__(self):
        for value in self.values:
            yield value

    def to_bytes(self, length, byteorder):
        return b''.join([v.to_bytes(length, byteorder) for v in self.values])

    def __repr__(self):
        return repr(self.values)

    def __eq__(self, other):
        return self.values == other.values

def get_class(arg, start=int):
    if isinstance(arg, (list, tuple, Vector)):
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

def zero_of_same_type(val):
    if isinstance(val, Vector):
        return Vector([zero_of_same_type(v) for v in val])
    else:
        return val.__class__(0)

def eval_poly_at(poly, pt):
    cls = get_class([poly, pt])
    o = zero_of_same_type(poly[0])
    power = cls(1)
    for coeff in poly:
        o += coeff * power
        power *= pt
    return o

def mul_polys(a, b):
    cls = get_class([a,b])
    o = [cls(0)] * (len(a) + len(b) - 1)
    for i, aval in enumerate(a):
        for j, bval in enumerate(b):
            o[i+j] += a[i] * b[j]
    return o

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

def multilinear_poly_eval(evals, pt):
    cls = get_class([evals, pt])
    assert len(evals) == 2 ** len(pt)
    o = cls(0)
    for i, evaluation in enumerate(evals):
        value = evals[i]
        for j, coord in enumerate(pt):
            if (i >> j) % 2:
                value *= coord
            else:
                value *= (cls(1) - coord)
        o += value
    return o

def extend(vals, expansion_factor=2):
    cls = get_class(vals)
    lagranges = [
        compute_lagrange_poly(len(vals), cls(i))
        for i in range(len(vals))
    ]
    output = vals[::]
    for x in range(len(vals), expansion_factor * len(vals)):
        _x = cls(x)
        o = zero_of_same_type(vals[0])
        for v, L in zip(vals, lagranges):
            o += v * eval_poly_at(L, x)
        output.append(o)
    return output

def evaluation_tensor_product(pt):
    cls = get_class(pt)
    o = [cls(1)]
    for coord in pt:
        o = [
            (cls(1) - coord) * v for v in o
        ] + [
            coord * v for v in o
        ]
    return o

