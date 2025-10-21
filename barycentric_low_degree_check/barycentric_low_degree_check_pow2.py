# A generalized low degree check based on Dankrad's check for degree < 2^l
from random import randint, shuffle, choice
from poly_utils import PrimeField

MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
PRIMITIVE_ROOT = 7

assert pow(PRIMITIVE_ROOT, (MODULUS - 1) // 2, MODULUS) != 1
assert pow(PRIMITIVE_ROOT, MODULUS - 1, MODULUS) == 1

primefield = PrimeField(MODULUS)

WIDTH = 32       # number of evaluations at roots of unity
M = 4            # check degree < M
N = WIDTH // M   # number of cosets

ROOT_OF_UNITY = pow(PRIMITIVE_ROOT, (MODULUS - 1) // WIDTH, MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(WIDTH)]

def check(f):
    r = randint(0, MODULUS)
    rm = pow(r, M, MODULUS)
    sums = [0] * N

    for i in range(WIDTH):
        coset_idx = i % N
        sums[coset_idx] += primefield.div(f[i] * DOMAIN[i], r - DOMAIN[i])

    for i in range(N):
        sums[i] = primefield.div(primefield.mul(sums[i], rm - DOMAIN[i * M]), DOMAIN[i * M])

    return sums.count(sums[0]) == N

fc = [randint(0, MODULUS) for i in range(M)]
f = [primefield.eval_poly_at(fc, x) for x in DOMAIN]

assert check(f)
f[randint(0, len(f) -1)] += 1
assert not check(f)