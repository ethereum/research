import blst
import hashlib
from poly_utils import PrimeField
from time import time
import sys
import gmpy2


#
# Proof of concept implementation for Eth1 simple custody
#
# https://notes.ethereum.org/1Rn2MwsoSWuEUHTnaRgLcw
#

# BLS12_381 curve modulus
MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

primefield = PrimeField(MODULUS)

# Proof of custody parameters
N = 15 # bits

DOMAIN = list(range(N))

def hash(x):
    if isinstance(x, bytes):
        return hashlib.sha256(x).digest()
    elif isinstance(x, blst.P1):
        return hash(x.compress())
    elif isinstance(x, int):
        return hash(x.to_bytes(32, "little"))
    b = b""
    for a in x:
        if isinstance(a, bytes):
            b += a
        elif isinstance(a, int):
            b += a.to_bytes(32, "little")
        elif isinstance(a, blst.P1):
            b += hash(a.compress())
    return hash(b)

C_CONSTANTS = [int.from_bytes(hash(i), "little") for i in range(N)]
D_CONSTANTS = [int.from_bytes(hash(i + N), "little") for i in range(N)]

def legendre(x):
    return gmpy2.jacobi(x, MODULUS)

def mod_sqrt(a):
    assert legendre(a) == 1

    # Factor p-1 on the form q * 2^s (with Q odd)
    q, s = MODULUS - 1, 0
    while q % 2 == 0:
        s += 1
        q //= 2

    # Select a z which is a quadratic non residue
    z = 1
    while legendre(z) != -1:
        z += 1
    c = pow(z, q, MODULUS)

    # Search for a solution
    x = pow(a, (q + 1) // 2, MODULUS)
    t = pow(a, q, MODULUS)
    m = s
    while t != 1:
        # Find the lowest i such that t^(2^i) = 1
        i, e = 0, 2
        for i in range(1, m):
            if pow(t, e, MODULUS) == 1:
                break
            e *= 2

        # Update next value to iterate
        b = pow(c, 2**(m - i - 1), MODULUS)
        x = (x * b) % MODULUS
        t = (t * b * b) % MODULUS
        c = (b * b) % MODULUS
        m = i

    assert (x ** 2 - a) % MODULUS == 0
    return x

def is_valid_custody_value(secret_key, custody_value):
    for i in range(N):
        if legendre(secret_key + C_CONSTANTS[i] * custody_value + D_CONSTANTS[i]) != 1:
            return False
    return True

def lincomb_naive(group_elements, factors, start_value = blst.G1().mult(0)):
    """
    Direct linear combination
    """
    assert len(group_elements) == len(factors)
    result = start_value.dup()
    for g, f in zip(group_elements, factors):
        result.add(g.dup().mult(f))
    return result

def generate_setup(N, secret):
    """
    Generates a setup in the G1 group and G2 group, as well as the Lagrange polynomials in G1 (via FFT)
    """
    g1_setup = [blst.G1().mult(pow(secret, i, MODULUS)) for i in range(N + 1)]
    g2_setup = [blst.G2().mult(pow(secret, i, MODULUS)) for i in range(N + 1)]
    lagrange_polys = primefield.lagrange_polys(list(range(N)))
    g1_lagrange = [lincomb_naive(g1_setup[:N], p) for p in lagrange_polys]
    g2_lagrange = [lincomb_naive(g2_setup[:N], p, start_value=blst.G2().mult(0)) for p in lagrange_polys]
    g2_zero = lincomb_naive(g2_setup, primefield.zero_poly(list(range(N))), start_value=blst.G2().mult(0))
    g2_one = lincomb_naive(g2_lagrange, [1] * N, start_value=blst.G2().mult(0))
    return {"g1": g1_setup, "g2": g2_setup, "g1_lagrange": g1_lagrange, "g2_zero": g2_zero, "g2_one": g2_one}

def compute_proof(setup, secret_key, custody_value):
    values = [secret_key + C_CONSTANTS[i] * custody_value + D_CONSTANTS[i] for i in range(N)]
    square_roots = [mod_sqrt(value) for value in values]
    d = primefield.lagrange_interp(list(range(N)), square_roots)
    D = lincomb_naive(setup["g1"][:N], d)
    E = lincomb_naive(setup["g2"][:N], d, start_value=blst.G2().mult(0))

    q = primefield.div_polys(primefield.mul_polys(d, d), primefield.zero_poly(list(range(N))))
    Pi = lincomb_naive(setup["g1"][:N - 1], q)
    return D.compress(), E.compress(), Pi.compress()

def check_proof_simple(setup, public_key_serialized, custody_value, proof):
    D_serialized, E_serialized, Pi_serialized = proof
    D = blst.P1(D_serialized)
    E = blst.P2(E_serialized)
    Pi = blst.P1(Pi_serialized)
    public_key = blst.P1(public_key_serialized)

    b_values = [C_CONSTANTS[i] * custody_value + D_CONSTANTS[i] for i in range(N)]
    B = lincomb_naive(setup["g1_lagrange"], b_values)
    C = public_key.dup().add(B)

    pairing = blst.PT(blst.G2().to_affine(), D.to_affine())
    pairing.mul(blst.PT(E.to_affine(), blst.G1().neg().to_affine()))
    if not pairing.final_exp().is_one():
        return False

    pairing = blst.PT(E.to_affine(), D.dup().neg().to_affine())
    pairing.mul(blst.PT(setup["g2_zero"].to_affine(), Pi.to_affine()))
    pairing.mul(blst.PT(blst.G2().to_affine(), C.to_affine()))
    if not pairing.final_exp().is_one():
        return False

    return True


def check_proof_combined(setup, public_key_serialized, custody_value, proof):
    D_serialized, E_serialized, Pi_serialized = proof
    D = blst.P1(D_serialized)
    E = blst.P2(E_serialized)
    Pi = blst.P1(Pi_serialized)

    r = int.from_bytes(hash(list(proof) + [public_key_serialized, custody_value]), "little") % MODULUS
    r2 = r * r % MODULUS

    public_key = blst.P1(public_key_serialized)

    b_values = [C_CONSTANTS[i] * custody_value + D_CONSTANTS[i] for i in range(N)]
    B = lincomb_naive(setup["g1_lagrange"], b_values)
    C = public_key.dup().add(B)

    pairing = blst.PT(blst.G2().mult(r).add(E).to_affine(), D.dup().neg().to_affine())
    pairing.mul(blst.PT(E.to_affine(), blst.G1().mult(r).to_affine()))
    pairing.mul(blst.PT(blst.G2().to_affine(), C.to_affine()))
    pairing.mul(blst.PT(setup["g2_zero"].to_affine(), Pi.to_affine()))

    return pairing.final_exp().is_one()


def get_proof_size(proof):
    return sum(len(x) for x in proof)


if __name__ == "__main__":
    time_a = time()
    setup = generate_setup(N, 8927347823478352432985)
    time_b = time()

    print("Computed setup in {0:.3f} ms".format(1000*(time_b - time_a)), file=sys.stderr)

    secret_key = pow(523487, 253478, MODULUS) + 1
    public_key = blst.G1().mult(secret_key).compress()

    time_a = time()
    custody_value = 876354679
    values_tried = 1
    while not is_valid_custody_value(secret_key, custody_value):
        custody_value += 1
        values_tried += 1
    time_b = time()

    print("Found custody value in {0:.3f} ms after {1} tries".format(1000*(time_b - time_a), values_tried), file=sys.stderr)

    time_a = time()
    proof = compute_proof(setup, secret_key, custody_value)
    time_b = time()
    
    proof_size = get_proof_size(proof)
    
    print("Computed proof (size = {0} bytes) in {1:.3f} ms".format(proof_size, 1000*(time_b - time_a)), file=sys.stderr)

    time_a = time()
    assert check_proof_simple(setup, public_key, custody_value, proof)
    time_b = time()
    check_time = time_b - time_a

    print("Checked proof in {0:.3f} ms".format(1000*(time_b - time_a)), file=sys.stderr)

    time_a = time()
    assert check_proof_combined(setup, public_key, custody_value, proof)
    time_b = time()
    check_time = time_b - time_a

    print("Checked proof (optimized/combined pairing) in {0:.3f} ms".format(1000*(time_b - time_a)), file=sys.stderr)

