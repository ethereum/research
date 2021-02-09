# This is a modification to the Bach algorithm that output not just factored
# numbers, but their moultiplicative group as well.
# The output is uniform in (N/2, N]

import multiprocessing as mp
import gmpy2
import random
import primefac
from collections import defaultdict
from bach_random_factored_numbers import lg, delta_n, factor_N, prime_power, process_r, hash
import bach_random_factored_numbers
from time import time
import sys

output = mp.Queue()
random_state = gmpy2.random_state()

threshold_bits = 60

def factor_N_with_multiplicative_group(N):
    f = primefac.factorint(N)
    r = []
    phi_hint = {}
    for p, alpha in f.items():
        for i in range(alpha):
            r.append(p)
        phi_hint[p] = factor_N(p - 1)
    return sorted(r), phi_hint


def process_f_with_multiplicative_group(N):
    while True:
        j = gmpy2.mpz_random(random_state, int(lg(N))) + 1
        r = gmpy2.mpz_random(random_state, 2 ** j)
        if j < threshold_bits:
            # For small numbers, run the old process_f
            # Multiplicative group hint can be computed later
            r = gmpy2.mpz_random(random_state, 2 ** j)
            q = 2 ** j + r
            if q > N:
                continue
            p, alpha = prime_power(q)
            if not p:
                continue
            l = gmpy2.mpfr_random(random_state)
            if l < delta_n(N, p, alpha) * 2 ** int(lg(q)):
                return p, alpha, {}
        else:
            # Generate a random factored number between (2 ** j - 1) and (2 ** (j + 1) - 2)
            # This is so that we get the multiplicative group of q
            x, xf = process_r(2 ** (j + 1) - 2)
            q = x + 1
            if q > N:
                continue
            p, alpha = prime_power(q)
            if not p:
                continue
            l = gmpy2.mpfr_random(random_state)
            if l < delta_n(N, p, alpha) * 2 ** int(lg(q)):
                if alpha == 1:
                    phi_hint = xf
                else:
                    # This is a special case: in this case xf isn't the multiplicative group
                    # But luckily since p^alpha - 1 = (p - 1) * (1 + p + ... + p^(alpha - 1))
                    # we will still have all the factors to get the factorisation of p - 1
                    phi_hint = []
                    remaining_x = x
                    for f in xf:
                        if remaining_x % f == 0:
                            phi_hint.append(f)
                            remaining_x /= f
                            if remaining_x == 1:
                                break
                return p, alpha, {p: phi_hint}

def process_r_with_multiplicative_group(N):
    if N < 2**threshold_bits:
        x = gmpy2.mpz_random(random_state, (N + 1) // 2) + N // 2 + 1
        xf = factor_N(x)
        # No factorisation hints for small factors, we can easily generate them later
        return x, xf, {}
    while True:
        p, alpha, phi_hint = process_f_with_multiplicative_group(N)
        q = p ** alpha
        Nprime = int(N // q)
        y, yf, phi_hint2 = process_r_with_multiplicative_group(Nprime)
        x = y * q
        l = gmpy2.mpfr_random(random_state)
        if l < gmpy2.log(N // 2) / gmpy2.log(x):
            phi_hint.update(phi_hint2)
            return x, [p] * alpha + yf, phi_hint

def generate_random_factored_numbers_with_multiplicative_group_mp(n, seed, num):
    global random_state
    random_state = gmpy2.random_state(seed)
    for i in range(num):
        r = process_r_with_multiplicative_group(n)
        complete_multiplicative_group_hint(r[1], r[2])
        output.put(r)


def generate_random_factored_numbers_with_multiplicative_group(bits, procs, count):

    count_per_proc = count // procs

    processes = [mp.Process(target=generate_random_factored_numbers_with_multiplicative_group_mp, \
                            args=(gmpy2.mpz(2**bits), random.randint(1, 10**10), count_per_proc)) for x in range(procs)]

    for p in processes:
        p.start()

    remaining_num = count % procs
    generate_random_factored_numbers_with_multiplicative_group_mp(gmpy2.mpz(2**bits), random.randint(1, 10**10), remaining_num)

    results = []

    for i in range(count):
        results.append(output.get())

    for p in processes:
        p.join()

    return results


def prod(l):
    r = 1
    for x in l:
        r *= x
    return r


def check_multiplicative_group(x, xf, xphi):
    xfdict = defaultdict(int)
    for p in xf:
        xfdict[p] += 1
    
    group_order = 1
    for p, alpha in xfdict.items():
        group_order *= (p - 1) * p ** (alpha - 1)
    
    return group_order == prod(xphi)



def complete_multiplicative_group_hint(xf, phi_hint):
    for p in xf:
        if p not in phi_hint:
            phi_hint[p] = factor_N(p - 1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python bach_random_factored_numbers.py bits procs count")
        print(" bits: Number of bits (e.g. 512 generates numbers between 2^511 and 2^512)")
        print(" procs: Number of parallel processes")
        print(" count: Number of factored numbers to generate")
        #print(" threshold_bits: Max bits for number to use 'sample and factor' base case")
        sys.exit(1)

    bits = int(sys.argv[1])
    procs = int(sys.argv[2])
    count = int(sys.argv[3])
    #threshold_bits = int(sys.argv[4])

    bach_random_factored_numbers.threshold_bits = threshold_bits

    time_a = time()

    results = generate_random_factored_numbers_with_multiplicative_group(bits, procs, count)

    time_b = time()

    for r in results:
        print("({0}, {1}, {2})".format(r[0], [int(x) for x in sorted(r[1])], {int(p): [int(x) for x in sorted(h)] for p, h in r[2].items()}))
    
    print("{0}\t{1}".format("\t".join(sys.argv[1:]), time_b - time_a), file=sys.stderr)
