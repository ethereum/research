def inv(a, modulus):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % modulus, modulus
    while low > 1:
        r = high//low
        nm, new = hm-lm*r, high-low*r
        lm, low, hm, high = nm, new, lm, low
    return lm % modulus

def eval_poly_at(poly, x, modulus):
    o, p = 0, 1
    for coeff in poly:
        o += coeff * p
        p = (p * x % modulus)
    return o % modulus

def lagrange_interp_4(pieces, xs, modulus):
    x01, x02, x03, x12, x13, x23 = \
        xs[0] * xs[1], xs[0] * xs[2], xs[0] * xs[3], xs[1] * xs[2], xs[1] * xs[3], xs[2] * xs[3]
    eq0 = [-x12 * xs[3] % modulus, (x12 + x13 + x23), -xs[1]-xs[2]-xs[3], 1]
    eq1 = [-x02 * xs[3] % modulus, (x02 + x03 + x23), -xs[0]-xs[2]-xs[3], 1]
    eq2 = [-x01 * xs[3] % modulus, (x01 + x03 + x13), -xs[0]-xs[1]-xs[3], 1]
    eq3 = [-x01 * xs[2] % modulus, (x01 + x02 + x12), -xs[0]-xs[1]-xs[2], 1]
    e0 = eval_poly_at(eq0, xs[0], modulus)
    e1 = eval_poly_at(eq1, xs[1], modulus)
    e2 = eval_poly_at(eq2, xs[2], modulus)
    e3 = eval_poly_at(eq3, xs[3], modulus)
    e01 = e0 * e1
    e23 = e2 * e3
    invall = inv(e01 * e23, modulus)
    inv_y0 = pieces[0] * invall * e1 * e23 % modulus
    inv_y1 = pieces[1] * invall * e0 * e23 % modulus
    inv_y2 = pieces[2] * invall * e01 * e3 % modulus
    inv_y3 = pieces[3] * invall * e01 * e2 % modulus
    return [(eq0[i] * inv_y0 + eq1[i] * inv_y1 + eq2[i] * inv_y2 + eq3[i] * inv_y3) % modulus for i in range(4)]

def lagrange_interp_2(pieces, xs, modulus):
    eq0 = [-xs[1] % modulus, 1]
    eq1 = [-xs[0] % modulus, 1]
    e0 = eval_poly_at(eq0, xs[0], modulus)
    e1 = eval_poly_at(eq1, xs[1], modulus)
    invall = inv(e0 * e1, modulus)
    inv_y0 = pieces[0] * invall * e1
    inv_y1 = pieces[1] * invall * e0
    return [(eq0[i] * inv_y0 + eq1[i] * inv_y1) % modulus for i in range(2)]
