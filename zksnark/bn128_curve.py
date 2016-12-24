from bn128_field_elements import field_modulus, FQ, FQ2, FQ12

curve_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Curve is y**2 = x**3 + 3
b = FQ(3)
b2 = FQ2([3, 0])
b12 = FQ12([3] + [0] * 11) / FQ12([0] * 6 + [1] + [0] * 5)


G1 = (FQ(1), FQ(2))
# Second element corresponds to modsqrt(67) * i in our quadratic field representation
G2 = (FQ2([4, 0]), FQ2([16893045765507297706785249332518927989146279141265438554111591828131739815230L, 16469166999615883226695964867118064280147127342783597836693979910667010785192]))

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
        return multiply(double(pt), n / 2)
    else:
        return add(multiply(double(pt), int(n / 2)), pt)

# Check that the G1 curve works fine
assert add(add(double(G1), G1), G1) == double(double(G1))
assert double(G1) != G1
assert add(multiply(G1, 9), multiply(G1, 5)) == add(multiply(G1, 12), multiply(G1, 2))
assert multiply(G1, curve_order) is None

# Check that the G2 curve works fine
assert add(add(double(G2), G2), G2) == double(double(G2))
assert double(G2) != G2
assert add(multiply(G2, 9), multiply(G2, 5)) == add(multiply(G2, 12), multiply(G2, 2))
assert multiply(G2, 2 * field_modulus - curve_order) is not None

# "Twist" a point in E(FQ2) into a point in E(FQ12)
w = FQ12([0, 1] + [0] * 10)

def twist(pt):
    if pt is None:
        return None
    x, y = pt
    nx = FQ12([x.coeffs[0]] + [0] * 5 + [x.coeffs[1]] + [0] * 5)
    ny = FQ12([y.coeffs[0]] + [0] * 5 + [y.coeffs[1]] + [0] * 5)
    return (nx / w **2, ny / w**3)

# Check that the twist creates a point that is on the curve
assert is_on_curve(twist(G2), b12)

# Check that the G12 curve works fine

G12 = twist(G2)
assert add(add(double(G12), G12), G12) == double(double(G12))
assert double(G12) != G12
assert add(multiply(G12, 9), multiply(G12, 5)) == add(multiply(G12, 12), multiply(G12, 2))
