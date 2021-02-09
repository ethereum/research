# Generates random factored numbers using the method described by Bach
# (SIAM J COMPUT Vol 17 No 2 1988, How to generate factored random numbers)
# Outputs of the algorithm are uniformly distributed in the interval (N/2, N]

import multiprocessing as mp
import gmpy2
import random
import primefac

output = mp.Queue()
random_state = gmpy2.random_state()

threshold_bits = 60

def lg(x):
    return gmpy2.log(x) / gmpy2.log(2)

def delta_n(N, p, alpha):
    return gmpy2.log(p) / gmpy2.log(N) * hash(N // p ** alpha) / N

def factor_N(N):
    f = primefac.factorint(N)
    r = []
    for p, alpha in f.items():
        for i in range(alpha):
            r.append(p)
    return sorted(r)

def prime_power(x):
    if gmpy2.is_prime(x):
        return x, 1
    if x == 0 or x == 1:
        return None, None
    if not gmpy2.is_power(x):
        return None, None
    alpha = 2
    root, is_root = gmpy2.iroot(x, alpha)
    while not (is_root and not gmpy2.is_power(root)):
        alpha += 1
        root, is_root = gmpy2.iroot(x, alpha)
    if gmpy2.is_prime(root):
        return root, alpha
    return None, None

def hash(a):
    return int((a + 1) // 2)

def process_f(N):
    while True:
        j = gmpy2.mpz_random(random_state, int(lg(N))) + 1
        r = gmpy2.mpz_random(random_state, 2 ** j)
        q = 2 ** j + r
        if q > N:
            continue
        p, alpha = prime_power(q)
        if not p:
            continue
        l = gmpy2.mpfr_random(random_state)
        if l < delta_n(N, p, alpha) * 2 ** int(lg(q)):
            return p, alpha

def process_r(N):
    if N < 2**threshold_bits:
        x = gmpy2.mpz_random(random_state, (N + 1) // 2) + N // 2 + 1
        return x, factor_N(x)
    while True:
        p, alpha = process_f(N)
        q = p ** alpha
        Nprime = int(N // q)
        y, yf = process_r(Nprime)
        x = y * q
        l = gmpy2.mpfr_random(random_state)
        if l < gmpy2.log(N // 2) / gmpy2.log(x):
            return x, [p] * alpha + yf

def generate_random_factored_numbers_mp(n, seed, num):
    global random_state
    random_state = gmpy2.random_state(seed)
    for i in range(num):
        output.put(process_r(n))

def generate_random_factored_numbers(bits, procs, count):

    count_per_proc = count // procs

    processes = [mp.Process(target=generate_random_factored_numbers_mp, \
                            args=(gmpy2.mpz(2**bits), random.randint(1, 10**10), count_per_proc)) for x in range(procs)]

    for p in processes:
        p.start()

    remaining_num = count % procs
    generate_random_factored_numbers_mp(gmpy2.mpz(2**bits), random.randint(1, 10**10), remaining_num)

    results = []

    for i in range(count):
        results.append(output.get())

    for p in processes:
        p.join()

    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python bach_random_factored_numbers.py bits procs count")
        print(" bits: Number of bits (e.g. 512 generates numbers between 2^511 and 2^512)")
        print(" procs: Number of parallel processes")
        print(" count: Number of factored numbers to generate")
        sys.exit(1)

    bits = int(sys.argv[1])
    procs = int(sys.argv[2])
    count = int(sys.argv[3])

    results = generate_random_factored_numbers(bits, procs, count)

    for r in results:
        print("(", r[0], ",", [int(x) for x in sorted(r[1])], ")")
