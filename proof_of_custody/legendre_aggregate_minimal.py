from py_ecc.optimized_bls12_381 import normalize, curve_order
from py_ecc.bls.api import sign, signature_to_G2, hash_to_G2
from py_ecc.fields import bls12_381_FQ2 as FQ2
from timeit import default_timer as timer
import gmpy2
bytes_per_chunk = 512
bits_per_chunk = bytes_per_chunk * 8
subchunks_bytes = [52] * 2 + [51] * 8
q = FQ2.field_modulus

validator_privkey = 3**1000 % curve_order
DOMAIN_RANDAO = 6

#Placeholder: will be RANDAO at some epoch
period = 1
period_bytes = period.to_bytes(32, "little")

def chunkify(x):
    chunks = [x[i:i + bytes_per_chunk] for i in range(0, len(x), bytes_per_chunk)]
    chunks[-1] = chunks[-1].ljust(bytes_per_chunk, b"\0")
    return chunks

def subchunkify(x):
    subchunks = []
    start = 0
    for size in subchunks_bytes:
        subchunks.append(x[start:start + size])
        start += size
    return subchunks

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

def legendre_aggregate_chunk(x, s1, s2):
    subchunks = subchunkify(x)
    bits = []
    for i, subchunk in enumerate(subchunks):
        a = int.from_bytes(subchunk, "little")
        bits.append(jacobi_bit((s1 if i % 2 == 0 else s2) + a, q))
    return sum(bits) % 2

def legendre_aggregate(x, s1, s2):
    bits = [legendre_aggregate_chunk(chunk, s1, s2) for chunk in chunkify(x)]
    return jacobi_bit(int.from_bytes(bitarray_to_bytes(bits), "little"), q)

def jacobi_bit_mpz(a, n):
    return (gmpy2.jacobi(a, n) + 1) // 2

def legendre_aggregate_chunk_mpz(x, s1, s2):
    subchunks = subchunkify(x)
    bits = []
    for i, subchunk in enumerate(subchunks):
        a = int.from_bytes(subchunk, "little")
        bits.append(jacobi_bit_mpz((s1 if i % 2 == 0 else s2) + a, q))
    return sum(bits) % 2

def legendre_aggregate_mpz(x, s1, s2):
    bits = [legendre_aggregate_chunk_mpz(chunk, s1, s2) for chunk in chunkify(x)]
    return jacobi_bit_mpz(int.from_bytes(bitarray_to_bytes(bits), "little"), q)

def legendre_aggregate_chunk_mpz_premul(x, s1, s2):
    subchunks = subchunkify(x)
    prod = gmpy2.mpz(1)
    for i, subchunk in enumerate(subchunks):
        a = int.from_bytes(subchunk, "little")
        prod *= gmpy2.mpz((s1 if i % 2 == 0 else s2) + a)
    return 1 - jacobi_bit_mpz(prod, q)

def legendre_aggregate_mpz_premul(x, s1, s2):
    bits = [legendre_aggregate_chunk_mpz_premul(chunk, s1, s2) for chunk in chunkify(x)]
    return jacobi_bit_mpz(int.from_bytes(bitarray_to_bytes(bits), "little"), q)

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

legendre_aggregate_mpz_premul(x,gmpy2.mpz(s1),gmpy2.mpz(s2))

time5 = timer()

assert [legendre_aggregate_chunk_mpz(chunk, s1, s2) for chunk in chunkify(x)] == [legendre_aggregate_chunk(chunk, s1, s2) for chunk in chunkify(x)]
assert [legendre_aggregate_chunk_mpz(chunk, s1, s2) for chunk in chunkify(x)] == [legendre_aggregate_chunk_mpz_premul(chunk, s1, s2) for chunk in chunkify(x)]

print("Time to sign = {0:.2f} s".format(time1 - time0))
print("Time to create test vector = {0:.2f} s".format(time2 - time1))
print("Time to Legendre aggregate (naive python) = {0:.2f} s".format(time3 - time2))
print("Time to Legendre aggregate (GMP) = {0:.2f} s".format(time4 - time3))
print("Time to Legendre aggregate (GMP premul) = {0:.2f} s".format(time5 - time4))