rsa_mod_size = 1024 # 1024 bit, roughly 80 bits of security
block_size = 2**19 # 512 kB block, current max
k = block_size // 32
r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001 # BLS curve order

import gmpy2
from timeit import default_timer as timer
ran = gmpy2.random_state()

prime_size = rsa_mod_size // 2

p = gmpy2.next_prime(gmpy2.mpz_random(ran,2**prime_size))
q = gmpy2.next_prime(gmpy2.mpz_random(ran,2**prime_size))
assert(gmpy2.gcd(p*q, (p-1)*(q-1)) == 1)
n = p * q
n2 = n**2
l = gmpy2.lcm(p-1, q-1)
g = gmpy2.mpz_random(ran, n2)
def L(x):
    return (x-1) // n
mu = gmpy2.powmod(L(gmpy2.powmod(g, l, n2)), -1, n)

def encrypt(x):
    """
    Paillier encryption
    """
    r = gmpy2.mpz_random(ran, n)
    return gmpy2.powmod(g, x, n2) * gmpy2.powmod(r, n, n2) % n2

def decrypt(y):
    """
    Paillier decryption
    """
    x = gmpy2.powmod(y, l, n2)
    return (L(x) * mu) % n

# Generate proof of custody secret
s = gmpy2.mpz_random(ran, r)

# Mock proof of custody data
d = [gmpy2.mpz_random(ran, r) for i in range(k)]

def compute_proof_of_custody(s, d):
    """
    Computes (simplified version of) UHF+Legendre Proof Of Custody
    """

    # Compute UHF
    uhf = 0
    spower = 1
    for d_i in d:
        uhf = (uhf + spower * d_i) % r
        spower = spower * s % r
    
    uhf = (uhf + spower) % r
        
    legendres = [gmpy2.jacobi(uhf + s + i, r) for i in range(10)]
    return uhf, legendres, all(x == 1 for x in legendres)

def compute_encrypted_key_powers(s, k):
    """
    Compute the powers of the custody key s, encrypted using Paillier. The validator
    (outsourcer) gives these to the provider so they can compute the proof of custody
    for them.
    """
    spower = 1
    enc_spowers = []
    for i in range(k + 2):
        enc_spowers.append(encrypt(spower))
        spower = spower * s % r
    return enc_spowers

def compute_encrypted_uhf(enc_spowers, d):
    """
    Computes the UHF result on a Paillier-encrypted custody key
    """

    enc_uhf = gmpy2.powmod(enc_spowers[0], d[0], n2)
    for enc_s_i, d_i in zip(enc_spowers[1:], d[1:]):
        enc_uhf = enc_uhf * gmpy2.powmod(enc_s_i, d_i, n2) % n2
    enc_uhf = enc_uhf * enc_spowers[len(d)] % n2
    return enc_uhf

def compute_proof_of_custody_from_uhf(s, uhf):
    """
    Computes proof of custody from UHF (only Legendre step)
    """
    legendres = [gmpy2.jacobi(uhf + s + i, r) for i in range(10)]
    return legendres, all(x == 1 for x in legendres)

time_a = timer()
uhf, legendres, proof_of_custody = compute_proof_of_custody(s, d)
time_b = timer()

print("Computed proof of custody (plaintext) in {0:.4f} s".format(time_b - time_a))

time_a = timer()
enc_spowers = compute_encrypted_key_powers(s, k)
time_b = timer()

print("Computed Paillier encrypted powers of custody key in {0:.4f} s".format(time_b - time_a))

time_a = timer()
encrypted_uhf = compute_encrypted_uhf(enc_spowers, d)
time_b = timer()

print("Computed encrypted UHF (outsourced computation) in {0:.4f} s".format(time_b - time_a))

time_a = timer()
decrypted_uhf = decrypt(encrypted_uhf) % r
decrypted_legendres, decrypted_poc = compute_proof_of_custody_from_uhf(s, uhf)
time_b = timer()

assert (uhf, legendres, proof_of_custody) == (decrypted_uhf, decrypted_legendres, decrypted_poc)

print("Decrypted UHF and computed proof of custody in {0:.4f} s".format(time_b - time_a))

