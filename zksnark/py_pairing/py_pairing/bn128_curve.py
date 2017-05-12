from bn128_field_elements import field_modulus, FQ
from optimized_field_elements import FQ2, FQ12
# from bn128_field_elements import FQ2, FQ12

curve_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Curve order should be prime
assert pow(2, curve_order, curve_order) == 2
# Curve order should be a factor of field_modulus**12 - 1
assert (field_modulus ** 12 - 1) % curve_order == 0

# Curve is y**2 = x**3 + 3
b = FQ(3)
# Twisted curve over FQ**2
b2 = FQ2([3, 0]) / FQ2([9, 1])
# Extension curve over FQ**12; same b value as over FQ
b12 = FQ12([3] + [0] * 11)

# Generator for curve over FQ
G1 = (FQ(1), FQ(2))
# Generator for twisted curve over FQ2
G2 = (FQ2([10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]),
      FQ2([8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]))

# Check that a point is on the curve defined by y**2 == x**3 + b
def is_on_curve(pt, b):
    if pt is None:
        return True
    x, y = pt
    return y**2 - x**3 == b

assert is_on_curve(G1, b)
assert is_on_curve(G2, b2)

# Elliptic curve doubling
def double(pt):
    x, y = pt
    l = 3 * x**2 / (2 * y)
    newx = l**2 - 2 * x
    newy = -l * newx + l * x - y
    return newx, newy

# Elliptic curve addition
def add(p1, p2):
    if p1 is None or p2 is None:
        return p1 if p2 is None else p2
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1 and y2 == y1:
        return double(p1)
    elif x2 == x1:
        return None
    else:
        l = (y2 - y1) / (x2 - x1)
    newx = l**2 - x1 - x2
    newy = -l * newx + l * x1 - y1
    assert newy == (-l * newx + l * x2 - y2)
    return (newx, newy)

# Elliptic curve point multiplication
def multiply(pt, n):
    if n == 0:
        return None
    elif n == 1:
        return pt
    elif not n % 2:
        return multiply(double(pt), n // 2)
    else:
        return add(multiply(double(pt), int(n // 2)), pt)

# Check that the G1 curve works fine
assert add(add(double(G1), G1), G1) == double(double(G1))
assert double(G1) != G1
assert add(multiply(G1, 9), multiply(G1, 5)) == add(multiply(G1, 12), multiply(G1, 2))
assert multiply(G1, curve_order) is None

# Check that the G2 curve works fine
assert add(add(double(G2), G2), G2) == double(double(G2))
assert double(G2) != G2
assert add(multiply(G2, 9), multiply(G2, 5)) == add(multiply(G2, 12), multiply(G2, 2))
assert multiply(G2, curve_order) is None
assert multiply(G2, 2 * field_modulus - curve_order) is not None
assert is_on_curve(multiply(G2, 9), b2)

# "Twist" a point in E(FQ2) into a point in E(FQ12)
w = FQ12([0, 1] + [0] * 10)

# Convert P => -P
def neg(pt):
    if pt is None:
        return None
    x, y = pt
    return (x, -y)

def twist(pt):
    if pt is None:
        return None
    _x, _y = pt
    # Field isomorphism from Z[p] / x**2 to Z[p] / x**2 - 18*x + 82
    xcoeffs = [_x.coeffs[0] - _x.coeffs[1] * 9, _x.coeffs[1]]
    ycoeffs = [_y.coeffs[0] - _y.coeffs[1] * 9, _y.coeffs[1]]
    # Isomorphism into subfield of Z[p] / w**12 - 18 * w**6 + 82,
    # where w**6 = x
    nx = FQ12([xcoeffs[0]] + [0] * 5 + [xcoeffs[1]] + [0] * 5)
    ny = FQ12([ycoeffs[0]] + [0] * 5 + [ycoeffs[1]] + [0] * 5)
    # Divide x coord by w**2 and y coord by w**3
    return (nx * w **2, ny * w**3)

# Check that the twist creates a point that is on the curve
assert is_on_curve(twist(G2), b12)

# Check that the G12 curve works fine

G12 = twist(G2)
assert add(add(double(G12), G12), G12) == double(double(G12))
assert double(G12) != G12
assert add(multiply(G12, 9), multiply(G12, 5)) == add(multiply(G12, 12), multiply(G12, 2))
assert is_on_curve(multiply(G12, 9), b12)
assert multiply(G12, curve_order) is None
