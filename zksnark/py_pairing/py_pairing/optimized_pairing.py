from .optimized_curve import double, add, multiply, is_on_curve, neg, twist, b, b2, b12, curve_order, G1, G2, G12, normalize
from .bn128_field_elements import field_modulus, FQ
from .optimized_field_elements import FQ2, FQ12

ate_loop_count = 29793968203157093288
log_ate_loop_count = 63
pseudo_binary_encoding = [0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0,
                          0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 1,
                          1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1,
                          1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, 1, 1]

assert sum([e * 2**i for i, e in enumerate(pseudo_binary_encoding)]) == ate_loop_count
                          

def normalize1(p):
    x, y = normalize(p)
    return x, y, x.__class__.one()

# Create a function representing the line between P1 and P2,
# and evaluate it at T. Returns a numerator and a denominator
# to avoid unneeded divisions
def linefunc(P1, P2, T):
    zero = P1[0].__class__.zero()
    x1, y1, z1 = P1
    x2, y2, z2 = P2
    xt, yt, zt = T
    # points in projective coords: (x / z, y / z)
    # hence, m = (y2/z2 - y1/z1) / (x2/z2 - x1/z1)
    # multiply numerator and denominator by z1z2 to get values below
    m_numerator = y2 * z1 - y1 * z2
    m_denominator = x2 * z1 - x1 * z2
    if m_denominator != zero:
        # m * ((xt/zt) - (x1/z1)) - ((yt/zt) - (y1/z1))
        return m_numerator * (xt * z1 - x1 * zt) - m_denominator * (yt * z1 - y1 * zt), \
            m_denominator * zt * z1
    elif m_numerator == zero:
        # m = 3(x/z)^2 / 2(y/z), multiply num and den by z**2
        m_numerator = 3 * x1 * x1
        m_denominator = 2 * y1 * z1
        return m_numerator * (xt * z1 - x1 * zt) - m_denominator * (yt * z1 - y1 * zt), \
            m_denominator * zt * z1
    else:
        return xt * z1 - x1 * zt, z1 * zt

def cast_point_to_fq12(pt):
    if pt is None:
        return None
    x, y, z = pt
    return (FQ12([x.n] + [0] * 11), FQ12([y.n] + [0] * 11), FQ12([z.n] + [0] * 11))

# Check consistency of the "line function"
one, two, three = G1, double(G1), multiply(G1, 3)
negone, negtwo, negthree = multiply(G1, curve_order - 1), multiply(G1, curve_order - 2), multiply(G1, curve_order - 3)

assert linefunc(one, two, one)[0] == FQ(0)
assert linefunc(one, two, two)[0] == FQ(0)
assert linefunc(one, two, three)[0] != FQ(0)
assert linefunc(one, two, negthree)[0] == FQ(0)
assert linefunc(one, negone, one)[0] == FQ(0)
assert linefunc(one, negone, negone)[0] == FQ(0)
assert linefunc(one, negone, two)[0] != FQ(0)
assert linefunc(one, one, one)[0] == FQ(0)
assert linefunc(one, one, two)[0] != FQ(0)
assert linefunc(one, one, negtwo)[0] == FQ(0)

# Main miller loop
def miller_loop(Q, P):
    if Q is None or P is None:
        return FQ12.one()
    R = Q
    f_num, f_den = FQ12.one(), FQ12.one()
    for b in pseudo_binary_encoding[63::-1]:
    #for i in range(log_ate_loop_count, -1, -1):
        _n, _d = linefunc(R, R, P)
        f_num = f_num * f_num * _n
        f_den = f_den * f_den * _d
        R = double(R)
        #if ate_loop_count & (2**i):
        if b == 1:
            _n, _d = linefunc(R, Q, P)
            f_num = f_num * _n
            f_den = f_den * _d
            R = add(R, Q)
        elif b == -1:
            nQ = neg(Q)
            _n, _d = linefunc(R, nQ, P)
            f_num = f_num * _n
            f_den = f_den * _d
            R = add(R, nQ)
    # assert R == multiply(Q, ate_loop_count)
    Q1 = (Q[0] ** field_modulus, Q[1] ** field_modulus, Q[2] ** field_modulus)
    # assert is_on_curve(Q1, b12)
    nQ2 = (Q1[0] ** field_modulus, -Q1[1] ** field_modulus, Q1[2] ** field_modulus)
    # assert is_on_curve(nQ2, b12)
    _n1, _d1 = linefunc(R, Q1, P)
    R = add(R, Q1)
    _n2, _d2 = linefunc(R, nQ2, P)
    f = f_num * _n1 * _n2 / (f_den * _d1 * _d2)
    # R = add(R, nQ2) This line is in many specifications but it technically does nothing
    return f ** ((field_modulus ** 12 - 1) // curve_order)

# Pairing computation
def pairing(Q, P):
    assert is_on_curve(Q, b2)
    assert is_on_curve(P, b)
    return miller_loop(twist(Q), cast_point_to_fq12(P))
