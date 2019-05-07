from py_ecc.optimized_bls12_381 import normalize, curve_order
from py_ecc.bls.api import sign, signature_to_G2, hash_to_G2
from py_ecc.fields import bls12_381_FQ2 as FQ2
from timeit import default_timer as timer
import gmpy2
bytes_per_legendre = 32
bits_per_legendre = bytes_per_legendre * 8
q = FQ2.field_modulus

validator_privkey = 3**1000 % curve_order
DOMAIN_RANDAO = 6

period = 1
period_bytes = period.to_bytes(32, "little")

def chunkify(x):
    chunks = [x[i:i + bytes_per_legendre] for i in range(0, len(x), bytes_per_legendre)]
    chunks[-1] = chunks[-1].ljust(bytes_per_legendre, b"\0")
    return chunks

def bitarray_to_bytes(x):
    c = 0
    ret = b""
    for i, b in enumerate(x):
        c = (c << 1) + b
        if i % 8 == 7:
            ret += c.to_bytes(1,"little")
            c = 0
    if i % 8 != 7:
        c <<= - len(x) % 8
        ret += c.to_bytes(1,"little")
    return ret

def test_vector(bytelength):
    ints = bytelength // 4
    return b"".join(i.to_bytes(4, "little") for i in range(ints))

def jacobi(a, n):
    if a > n:
        return jacobi(a % n, n)
    assert(n > a > 0 and n % 2 == 1)
    t = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            r = n % 8
            if r == 3 or r == 5:
                t = -t
        a, n = n, a
        if a % 4 == n % 4 == 3:
            t = -t
        a %= n
    if n == 1:
        return t
    else:
        return 0
    
def jacobi_bit(a, n):
    return (jacobi(a, n) + 1) // 2

def legendre_aggregate_round(x, s1, s2):
    chunks = chunkify(x)
    bits = []
    for chunk in chunks:
        a = int.from_bytes(chunk, "little")
        bits.append(jacobi_bit(s1 + a, q) ^ jacobi_bit(s2 + a, q))
    return bitarray_to_bytes(bits)

def legendre_aggregate(x, s1, s2):
    while len(x) > bytes_per_legendre:
        x = legendre_aggregate_round(x, s1, s2)
    x = legendre_aggregate_round(x, s1, s2)
    return x != b"\0"

def jacobi_bit_mpz(a, n):
    return (gmpy2.jacobi(a, n) + 1) // 2

def legendre_aggregate_round_mpz(x, s1, s2):
    chunks = chunkify(x)
    bits = []
    for chunk in chunks:
        a = gmpy2.mpz(int.from_bytes(chunk, "little"))
        bits.append(jacobi_bit_mpz(s1 + a, q) ^ jacobi_bit_mpz(s2 + a, q))
    return bitarray_to_bytes(bits)

def legendre_aggregate_mpz(x, s1, s2):
    while len(x) > bytes_per_legendre:
        x = legendre_aggregate_round_mpz(x, s1, s2)
    x = legendre_aggregate_round_mpz(x, s1, s2)
    return x != b"\0"

time0 = timer()

signature = normalize(signature_to_G2(sign(period_bytes, validator_privkey, DOMAIN_RANDAO)))

time1 = timer()

s1, s2 = signature[0].coeffs

x = test_vector(2**21)

time2 = timer()

legendre_aggregate(x,gmpy2.mpz(s1),gmpy2.mpz(s2))

time3 = timer()

legendre_aggregate_mpz(x,gmpy2.mpz(s1),gmpy2.mpz(s2))

time4 = timer()

assert legendre_aggregate_round_mpz(x, s1, s2) == legendre_aggregate_round(x, s1, s2)

print("Time to sign = {0:.2f} s".format(time1 - time0))
print("Time to create test vector = {0:.2f} s".format(time2 - time1))
print("Time to Legendre aggregate (naive python) = {0:.2f} s".format(time3 - time2))
print("Time to Legendre aggregate (GMP) = {0:.2f} s".format(time4 - time3))