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

