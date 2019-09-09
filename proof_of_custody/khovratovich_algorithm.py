# Implements the algorithm described in https://eprint.iacr.org/2019/862.pdf

import gmpy2
import math
from timeit import default_timer as timer
from collections import defaultdict

p = gmpy2.next_prime(2**40)
num_challenge_bits = 2 ** 20
legendre_evals = 0

def jacobi_bit_mpz(a, n):
    global legendre_evals
    legendre_evals += 1
    return (gmpy2.jacobi(a, n) + 1) // 2

# Create challenge
key = 3**1000 % p
challenge = [jacobi_bit_mpz(key + x, p) for x in range(num_challenge_bits)]

# Solve challenge using Khovratovich algorithm
logN1 = math.ceil(math.log2(len(challenge)))

N1 = len(challenge) - logN1


def bitstring_to_int(a):
    return sum(x*2**i for i, x in enumerate(a))


def find_match(challenge, p):
    
    # Create dictionary of all strings of logN1 bits in the challenge
    cdict = defaultdict(list)
    for i in range(N1):
        c = bitstring_to_int(challenge[i: i + logN1])
        cdict[c].append(i)

    
    current_key = 0
    expected_N2 = p // N1 // 2
    number_of_tries = 0
    while True:
        number_of_tries += 1
        if number_of_tries % 100000 == 0:
            print("Tried ", number_of_tries, "keys (expected = {0})".format(expected_N2))
        current_key = (current_key + N1) % p
        c = bitstring_to_int([jacobi_bit_mpz(current_key + x, p) for x in range(logN1)])
        if c in cdict:
            found_match = False
            for key_offset in cdict[c]:
                predicted_key = current_key - key_offset
                if all(jacobi_bit_mpz(predicted_key + x, p) == challenge[x] for x in range(math.ceil(math.log2(p) * 2))):
                    return predicted_key
            if found_match:
                break


start = timer()
legendre_evals = 0
assert find_match(challenge, p) == key
end = timer()
print("Time taken: {0:.2f} s".format(end - start))
print("Total Legendre evaluations: {0}".format(legendre_evals))