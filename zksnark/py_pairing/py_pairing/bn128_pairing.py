from bn128_curve import double, add, multiply, is_on_curve, neg, twist, b, b2, b12, curve_order, G1, G2, G12
from bn128_field_elements import field_modulus, FQ
from optimized_field_elements import FQ2, FQ12

ate_loop_count = 29793968203157093288
log_ate_loop_count = 63

# Create a function representing the line between P1 and P2,
# and evaluate it at T
def linefunc(P1, P2, T):
    assert P1 and P2 and T # No points-at-infinity allowed, sorry
    x1, y1 = P1
    x2, y2 = P2
    xt, yt = T
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        return m * (xt - x1) - (yt - y1)
    elif y1 == y2:
        m = 3 * x1**2 / (2 * y1)
        return m * (xt - x1) - (yt - y1)
    else:
        return xt - x1

def cast_point_to_fq12(pt):
    if pt is None:
        return None
    x, y = pt
    return (FQ12([x.n] + [0] * 11), FQ12([y.n] + [0] * 11))

# Check consistency of the "line function"
one, two, three = G1, double(G1), multiply(G1, 3)
negone, negtwo, negthree = multiply(G1, curve_order - 1), multiply(G1, curve_order - 2), multiply(G1, curve_order - 3)

assert linefunc(one, two, one) == FQ(0)
assert linefunc(one, two, two) == FQ(0)
assert linefunc(one, two, three) != FQ(0)
assert linefunc(one, two, negthree) == FQ(0)
assert linefunc(one, negone, one) == FQ(0)
assert linefunc(one, negone, negone) == FQ(0)
assert linefunc(one, negone, two) != FQ(0)
assert linefunc(one, one, one) == FQ(0)
assert linefunc(one, one, two) != FQ(0)
assert linefunc(one, one, negtwo) == FQ(0)

# Main miller loop
def miller_loop(Q, P):
    if Q is None or P is None:
        return FQ12.one()
    R = Q
    f = FQ12.one()
    for i in range(log_ate_loop_count, -1, -1):
        f = f * f * linefunc(R, R, P)
        R = double(R)
        if ate_loop_count & (2**i):
            f = f * linefunc(R, Q, P)
            R = add(R, Q)
    # assert R == multiply(Q, ate_loop_count)
    Q1 = (Q[0] ** field_modulus, Q[1] ** field_modulus)
    # assert is_on_curve(Q1, b12)
    nQ2 = (Q1[0] ** field_modulus, -Q1[1] ** field_modulus)
    # assert is_on_curve(nQ2, b12)
    f = f * linefunc(R, Q1, P)
    R = add(R, Q1)
    f = f * linefunc(R, nQ2, P)
    # R = add(R, nQ2) This line is in many specifications but it technically does nothing
    return f ** ((field_modulus ** 12 - 1) // curve_order)

# Pairing computation
def pairing(Q, P):
    assert is_on_curve(Q, b2)
    assert is_on_curve(P, b)
    return miller_loop(twist(Q), cast_point_to_fq12(P))
