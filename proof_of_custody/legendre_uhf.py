from py_ecc.optimized_bls12_381 import normalize, curve_order
from py_ecc.bls.api import sign, signature_to_G2, hash_to_G2
from py_ecc.fields import bls12_381_FQ2 as FQ2
from timeit import default_timer as timer
import gmpy2
BYTES_PER_CUSTODY_CHUNK = 48
BITS_PER_CUSTODY_CHUNK = BYTES_PER_CUSTODY_CHUNK * 8
BLS12_381_Q = FQ2.field_modulus

validator_privkey = 3**1000 % curve_order
DOMAIN_RANDAO = (6).to_bytes(4, "little")

#Placeholder: will be RANDAO at some epoch
period = 1
period_bytes = period.to_bytes(32, "little")

def custody_chunkify(bytez: bytes) -> list:
    bytez += b'\x00' * (-len(bytez) % BYTES_PER_CUSTODY_CHUNK)
    return [bytez[i:i + BYTES_PER_CUSTODY_CHUNK]
            for i in range(0, len(bytez), BYTES_PER_CUSTODY_CHUNK)]

def test_vector(bytelength):
    ints = bytelength // 4
    return b"".join(i.to_bytes(4, "little") for i in range(ints))

def legendre_bit(a: int, q: int) -> bool:
    if a >= q:
        return legendre_bit(a % q, q)
    if a == 0:
        return True
    assert(q > a > 0 and q % 2 == 1)
    t = True
    n = q
    while a != 0:
        while a % 2 == 0:
            a //= 2
            r = n % 8
            if r == 3 or r == 5:
                t = not t
        a, n = n, a
        if a % 4 == n % 4 == 3:
            t = not t
        a %= n
    if n == 1:
        return t
    else:
        return True

def get_polynomial_uhf(chunks: list, key: int) -> int:
    r = 1
    for chunk in reversed(chunks):
        r *= key
        r += int.from_bytes(chunk, 'little')
        r %= BLS12_381_Q
    return r

def get_legendre_prf(data: int, key: int) -> bool:
    return legendre_bit(key + data, BLS12_381_Q)

def get_legendre_uhf(data: bytes, key1: int, key2: int) -> bool:
    return get_legendre_prf(get_polynomial_uhf(custody_chunkify(data), key1), key2)

def legendre_bit_mpz(a, n):
    return True if gmpy2.jacobi(a, n) >= 0 else False

def get_polynomial_uhf_mpz(chunks: list, key: int) -> int:
    r = 1
    q = gmpy2.mpz(BLS12_381_Q)
    for chunk in reversed(chunks):
        r *= key
        r += gmpy2.mpz(int.from_bytes(chunk, 'little'))
        r %= q
    return r

def get_legendre_prf_mpz(data: int, key: int) -> bool:
    return legendre_bit_mpz(key + data, BLS12_381_Q)

def get_legendre_uhf_mpz(data: bytes, key1: int, key2: int) -> bool:
    return get_legendre_prf_mpz(get_polynomial_uhf_mpz(custody_chunkify(data), key1), key2)

time0 = timer()

signature = normalize(signature_to_G2(sign(period_bytes, validator_privkey, DOMAIN_RANDAO)))

time1 = timer()

key1, key2 = signature[0].coeffs

data = test_vector(2**17)

time2 = timer()

for i in range(1000):
    get_legendre_uhf(data, key1, key2)

time3 = timer()

for i in range(1000):
    get_legendre_uhf_mpz(data, gmpy2.mpz(key1), gmpy2.mpz(key2))

time4 = timer()

assert get_polynomial_uhf(custody_chunkify(data), key1) == get_polynomial_uhf_mpz(custody_chunkify(data), key1)
assert get_legendre_uhf(data, key1, key2) == get_legendre_uhf_mpz(data, gmpy2.mpz(key1), gmpy2.mpz(key2))

print("Time to sign = {0:.4f} s".format(time1 - time0))
print("Time to create test vector = {0:.4f} s".format(time2 - time1))
print("Time to Legendre UHF (naive python) = {0:.4f} s".format((time3 - time2) / 1000))
print("Time to Legendre UHF (GMP) = {0:.4f} s".format((time4 - time3) / 1000 ))
