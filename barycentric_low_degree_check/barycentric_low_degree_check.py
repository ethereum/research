from random import randint, shuffle, choice
from poly_utils import PrimeField

MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
PRIMITIVE_ROOT = 7

assert pow(PRIMITIVE_ROOT, (MODULUS - 1) // 2, MODULUS) != 1
assert pow(PRIMITIVE_ROOT, MODULUS - 1, MODULUS) == 1

primefield = PrimeField(MODULUS)

WIDTH = 8

ROOT_OF_UNITY = pow(PRIMITIVE_ROOT, (MODULUS - 1) // WIDTH, MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(WIDTH)]

def check(f):
    result = 0
    r = randint(0, MODULUS)
    rn2 = pow(r, WIDTH // 2, MODULUS)

    for i in range(WIDTH):
        summand = f[i] * ((-1)**i * rn2 - 1)
        summand = primefield.div(summand, DOMAIN[i * (WIDTH // 2 - 1) % WIDTH] * (r - DOMAIN[i]))
        result += summand

    return result % MODULUS

fc = [randint(0, MODULUS) for i in range(4)]
f = [primefield.eval_poly_at(fc, x) for x in DOMAIN]

print(check(f))