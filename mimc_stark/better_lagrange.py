def inv(a, modulus):
    return pow(a, (modulus - 2), modulus)

def eval_poly_at(poly, x, modulus):
    o, p = 0, 1
    for coeff in poly:
        o += coeff * p
        p = (p * x % modulus)
    return o % modulus

