from py_ecc import optimized_bls12_381 as b
from fft import fft
from multicombs import lincomb

# Generatore for the field
PRIMITIVE_ROOT = 5
MODULUS = b.curve_order

assert pow(PRIMITIVE_ROOT, (MODULUS - 1) // 2, MODULUS) != 1
assert pow(PRIMITIVE_ROOT, MODULUS - 1, MODULUS) == 1

#########################################################################################
#
# Helpers
#
#########################################################################################

def is_power_of_two(x):
    return x > 0 and x & (x-1) == 0


def generate_setup(s, size):
    """
    # Generate trusted setup, in coefficient form.
    # For data availability we always need to compute the polynomials anyway, so it makes little sense to do things in Lagrange space
    """
    return (
        [b.multiply(b.G1, pow(s, i, MODULUS)) for i in range(size + 1)],
        [b.multiply(b.G2, pow(s, i, MODULUS)) for i in range(size + 1)],
    )

#########################################################################################
#
# Field operations
#
#########################################################################################


def get_root_of_unity(order):
    """
    Returns a root of unity of order "order"
    """
    assert (MODULUS - 1) % order == 0
    return pow(PRIMITIVE_ROOT, (MODULUS - 1) // order, MODULUS)

def inv(a):
    """
    Modular inverse using eGCD algorithm
    """
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % MODULUS, MODULUS
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % MODULUS

def div(x, y):
    return x * inv(y) % MODULUS

#########################################################################################
#
# Polynomial operations
#
#########################################################################################

def eval_poly_at(p, x):
    """
    Evaluate polynomial p (coefficient form) at point x
    """
    y = 0
    power_of_x = 1
    for i, p_coeff in enumerate(p):
        y += power_of_x * p_coeff
        power_of_x = (power_of_x * x) % MODULUS
    return y % MODULUS

def div_polys(a, b):
    """
    Long polynomial difivion for two polynomials in coefficient form
    """
    a = [x for x in a]
    o = []
    apos = len(a) - 1
    bpos = len(b) - 1
    diff = apos - bpos
    while diff >= 0:
        quot = div(a[apos], b[bpos])
        o.insert(0, quot)
        for i in range(bpos, -1, -1):
            a[diff + i] -= b[i] * quot
        apos -= 1
        diff -= 1
    return [x % MODULUS for x in o]

#########################################################################################
#
# Utils for reverse bit order
#
#########################################################################################


def reverse_bit_order(n, order):
    """
    Reverse the bit order of an integer n
    """
    assert is_power_of_two(order)
    # Convert n to binary with the same number of bits as "order" - 1, then reverse its bit order
    return int(('{:0' + str(order.bit_length() - 1) + 'b}').format(n)[::-1], 2)
    

def list_to_reverse_bit_order(l):
    """
    Convert a list between normal and reverse bit order. This operation is idempotent.
    """
    return [l[reverse_bit_order(i, len(l))] for i in range(len(l))]


#########################################################################################
#
# Converting between polynomials (in coefficient form) and data (in reverse bit order)
# and extending data
#
#########################################################################################

def get_polynomial(data):
    """
    Interpolate a polynomial (coefficients) from data in reverse bit order
    """
    assert is_power_of_two(len(data))
    root_of_unity = get_root_of_unity(len(data))
    return fft(list_to_reverse_bit_order(data), MODULUS, root_of_unity, True)

def get_data(polynomial):
    """
    Get data (in reverse bit order) from polynomial in coefficient form
    """
    assert is_power_of_two(len(polynomial))
    root_of_unity = get_root_of_unity(len(polynomial))
    return list_to_reverse_bit_order(fft(polynomial, MODULUS, root_of_unity, False))

def get_extended_data(polynomial):
    """
    Get extended data (expanded by 2x, reverse bit order) from polynomial in coefficient form
    """
    assert is_power_of_two(len(polynomial))
    extended_polynomial = polynomial + [0] * len(polynomial)
    root_of_unity = get_root_of_unity(len(extended_polynomial))
    return list_to_reverse_bit_order(fft(extended_polynomial, MODULUS, root_of_unity, False))

#########################################################################################
#
# Kate single proofs
#
#########################################################################################

def commit_to_poly(polynomial, setup):
    """
    Kate commitment to polynomial in coefficient form
    """
    return lincomb(setup[0][:len(polynomial)], polynomial, b.add, b.Z1)

def compute_proof_single(polynomial, x, setup):
    """
    Compute Kate proof for polynomial in coefficient form at position x
    """
    quotient_polynomial = div_polys(polynomial, [-x, 1])
    return lincomb(setup[0][:len(quotient_polynomial)], quotient_polynomial, b.add, b.Z1)

def check_proof_single(commitment, proof, x, y, setup):
    """
    Check a proof for a Kate commitment for an evaluation f(x) = y
    """
    # Verify the pairing equation
    #
    # e([commitment - y], [1]) = e([proof],  [s - x])
    #    equivalent to
    # e([commitment - y]^(-1), [1]) * e([proof],  [s - x]) = 1_T
    #

    s_minus_x = b.add(setup[1][1], b.multiply(b.neg(b.G2), x))
    commitment_minus_y = b.add(commitment, b.multiply(b.neg(b.G1), y))

    pairing_check = b.pairing(b.G2, b.neg(commitment_minus_y), False)
    pairing_check *= b.pairing(s_minus_x, proof, False)
    pairing = b.final_exponentiate(pairing_check)

    return pairing == b.FQ12.one()

#########################################################################################
#
# Kate multiproofs on a coset
#
#########################################################################################

def compute_proof_multi(polynomial, x, n, setup):
    """
    Compute Kate proof for polynomial in coefficient form at positions x * w^y where w is
    an n-th root of unity (this is the proof for one data availability sample, which consists
    of several polynomial evaluations)
    """
    quotient_polynomial = div_polys(polynomial, [-pow(x, n, MODULUS)] + [0] * (n - 1) + [1])
    return lincomb(setup[0][:len(quotient_polynomial)], quotient_polynomial, b.add, b.Z1)

def check_proof_multi(commitment, proof, x, ys, setup):
    """
    Check a proof for a Kate commitment for an evaluation f(x w^i) = y_i
    """
    n = len(ys)
    root_of_unity = get_root_of_unity(n)
    
    # Interpolate at a coset. Note because it is a coset, not the subgroup, we have to multiply the
    # polynomial coefficients by x^i
    interpolation_polynomial = fft(ys, MODULUS, root_of_unity, True)
    interpolation_polynomial = [div(c, pow(x, i, MODULUS)) for i, c in enumerate(interpolation_polynomial)]

    # Verify the pairing equation
    #
    # e([commitment - interpolation_polynomial(s)], [1]) = e([proof],  [s^n - x^n])
    #    equivalent to
    # e([commitment - interpolation_polynomial]^(-1), [1]) * e([proof],  [s^n - x^n]) = 1_T
    #

    xn_minus_yn = b.add(setup[1][n], b.multiply(b.neg(b.G2), pow(x, n, MODULUS)))
    commitment_minus_interpolation = b.add(commitment, b.neg(lincomb(
        setup[0][:len(interpolation_polynomial)], interpolation_polynomial, b.add, b.Z1)))
    pairing_check = b.pairing(b.G2, b.neg(commitment_minus_interpolation), False)
    pairing_check *= b.pairing(xn_minus_yn, proof, False)
    pairing = b.final_exponentiate(pairing_check)
    return pairing == b.FQ12.one()

if __name__ == "__main__":
    polynomial = [1, 2, 3, 4, 7, 7, 7, 7, 13, 13, 13, 13, 13, 13, 13, 13]
    n = len(polynomial)

    setup = generate_setup(1927409816240961209460912649124, n)

    commitment = commit_to_poly(polynomial, setup)
    proof = compute_proof_single(polynomial, 17, setup)
    value = eval_poly_at(polynomial, 17)
    assert check_proof_single(commitment, proof, 17, value, setup)
    print("Single point check passed")

    root_of_unity = get_root_of_unity(8)
    x = 5431
    coset = [x * pow(root_of_unity, i, MODULUS) for i in range(8)]
    ys = [eval_poly_at(polynomial, z) for z in coset]
    proof = compute_proof_multi(polynomial, x, 8, setup)
    assert check_proof_multi(commitment, proof, x, ys, setup)
    print("Coset check passed")