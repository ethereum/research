import blst
from fft import fft
from multicombs import lincomb

# Generatore for the field
PRIMITIVE_ROOT = 5
MODULUS = 52435875175126190479447740508185965837690552500527637822603658699938581184513 #b.curve_order

P1_INF = blst.P1(bytes.fromhex("400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))
P2_INF = blst.P2(bytes.fromhex("400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))

assert pow(PRIMITIVE_ROOT, (MODULUS - 1) // 2, MODULUS) != 1
assert pow(PRIMITIVE_ROOT, MODULUS - 1, MODULUS) == 1

#########################################################################################
#
# Helpers
#
#########################################################################################
def hex(i : blst.P1):
    bytes = i.serialize()
    return ''.join(f'{b:02x}' for b in bytes)

def is_power_of_two(x):
    return x > 0 and x & (x-1) == 0


def generate_setup(s :int, size :int) -> tuple[list[blst.P1], list[blst.P2]]:
    """
    # Generate trusted setup, in coefficient form.
    # For data availability we always need to compute the polynomials anyway, so it makes little sense to do things in Lagrange space
    """
    return(
        [blst.G1().mult( pow(s, i, MODULUS) ) for i in range(size + 1)],
        [blst.G2().mult( pow(s, i, MODULUS) ) for i in range(size + 1)]
    )

#########################################################################################
#
# Field operations
#
#########################################################################################


def get_root_of_unity(order: int) -> int:
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

def eval_poly_at(p: list[int], x: int) -> int:
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

BLSTADDER = lambda x,y: x.dup().add(y)

def commit_to_poly(polynomial: list[int], setup: tuple[list[blst.P1], list[blst.P2]]) -> blst.P1:
    """
    Kate commitment to polynomial in coefficient form
    """
    return lincomb(setup[0][:len(polynomial)], polynomial, BLSTADDER, P1_INF.dup())

def compute_proof_single(polynomial: list[int], x: int, setup: tuple[list[blst.P1], list[blst.P2]]) -> blst.P1:
    """
    Compute Kate proof for polynomial in coefficient form at position x
    """
    quotient_polynomial = div_polys(polynomial, [-x, 1])
    return lincomb(setup[0][:len(quotient_polynomial)], quotient_polynomial, BLSTADDER, P1_INF.dup())

def check_proof_single(commitment: blst.P1, proof: blst.P1, x: int, y: int, setup: tuple[list[blst.P1], list[blst.P2]]) -> bool:
    """
    Check a proof for a Kate commitment for an evaluation f(x) = y
    """
    # Verify the pairing equation
    #
    # e([commitment - y], [1]) = e([proof],  [s - x])
    #    equivalent to
    # e([commitment - y]^(-1), [1]) * e([proof],  [s - x]) = 1_T
    #

    smx = setup[1][1].dup().add(blst.G2().neg().mult(x))
    cmy = commitment.dup().add(blst.G1().neg().mult(y))

    pc = blst.PT(blst.G2().to_affine(), cmy.neg().to_affine())
    pc.mul(blst.PT(smx.to_affine(), proof.to_affine()))
    return pc.final_exp().is_one()

#########################################################################################
#
# Kate multiproofs on a coset
#
#########################################################################################

def compute_proof_multi(polynomial: list[int], x: blst.P1, n:int , setup: blst.P1) -> blst.P1:
    """
    Compute Kate proof for polynomial in coefficient form at positions x * w^y where w is
    an n-th root of unity (this is the proof for one data availability sample, which consists
    of several polynomial evaluations)
    """
    quotient_polynomial = div_polys(polynomial, [-pow(x, n, MODULUS)] + [0] * (n - 1) + [1])
    return lincomb(setup[0][:len(quotient_polynomial)], quotient_polynomial, BLSTADDER, P1_INF.dup())

def check_proof_multi(commitment: blst.P1, proof: blst.P1, x: int, ys: list[int], setup: tuple[list[blst.P1], list[blst.P2]]) ->bool:
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

    #xn_minus_yn = b.add(setup[1][n], b.multiply(b.neg(b.G2), pow(x, n, MODULUS)))
    xn_minus_yn = setup[1][n].dup().add( blst.G2().neg().mult(pow(x, n, MODULUS)) )
    #commitment_minus_interpolation = b.add(commitment, b.neg(lincomb(
    #    setup[0][:len(interpolation_polynomial)], interpolation_polynomial, b.add, b.Z1)))
    commitment_minus_interpolation = commitment.dup().add( lincomb(
        setup[0][:len(interpolation_polynomial)], interpolation_polynomial, BLSTADDER, P1_INF.dup()
        ).neg() ) 

    pc = blst.PT(blst.G2().to_affine(), commitment_minus_interpolation.neg().to_affine())
    pc.mul(blst.PT(xn_minus_yn.to_affine(), proof.to_affine()))
    return pc.final_exp().is_one()

if __name__ == "__main__":
    from timer import chrono
    import random
    from tqdm import tqdm
    # Uncomment to report time of the functions
    #generate_setup = chrono(generate_setup)
    #commit_to_poly = chrono(commit_to_poly)
    #eval_poly_at = chrono(eval_poly_at)
    #check_proof_single = chrono(check_proof_single)
    #compute_proof_multi = chrono(compute_proof_multi)
    #check_proof_multi = chrono(check_proof_multi)

    MAX_DEGREE_POLY = MODULUS-1
    N_POINTS = 512
    polynomial = [random.randint(1, MAX_DEGREE_POLY) for _ in range(N_POINTS)] 
    n = N_POINTS

    setup = generate_setup(random.getrandbits(256), n)

    commitment = commit_to_poly(polynomial, setup)
    
    for i in tqdm(range(n), desc='Single point test'):
        proof = compute_proof_single(polynomial, i, setup)
        value = eval_poly_at(polynomial, i)
        assert check_proof_single(commitment, proof, i, value, setup)
    print(f"Single point check passed for all {n} points")

    root_of_unity = get_root_of_unity(8)
    x = 5431
    coset = [x * pow(root_of_unity, i, MODULUS) for i in range(8)]
    ys = [eval_poly_at(polynomial, z) for z in coset]
    proof = compute_proof_multi(polynomial, x, 8, setup)
    assert check_proof_multi(commitment, proof, x, ys, setup)
    print("Coset check passed")