import time

def legendre_symbol(a, p):
    """
    Legendre symbol
    Define if a is a quadratic residue modulo odd prime
    http://en.wikipedia.org/wiki/Legendre_symbol
    """
    ls = pow(a, (p - 1)/2, p)
    if ls == p - 1:
        return -1
    return ls

def prime_mod_sqrt(a, p):
    """
    Square root modulo prime number
    Solve the equation
        x^2 = a mod p
    and return list of x solution
    http://en.wikipedia.org/wiki/Tonelli-Shanks_algorithm
    """
    a %= p

    # Simple case
    if a == 0:
        return [0]
    if p == 2:
        return [a]

    # Check solution existence on odd prime
    if legendre_symbol(a, p) != 1:
        return []

    # Simple case
    if p % 4 == 3:
        x = pow(a, (p + 1)/4, p)
        return [x, p-x]

    # Factor p-1 on the form q * 2^s (with Q odd)
    q, s = p - 1, 0
    while q % 2 == 0:
        s += 1
        q //= 2

    # Select a z which is a quadratic non resudue modulo p
    z = 1
    while legendre_symbol(z, p) != -1:
        z += 1
    c = pow(z, q, p)

    # Search for a solution
    x = pow(a, (q + 1)/2, p)
    t = pow(a, q, p)
    m = s
    while t != 1:
        # Find the lowest i such that t^(2^i) = 1
        i, e = 0, 2
        for i in xrange(1, m):
            if pow(t, e, p) == 1:
                break
            e *= 2

        # Update next value to iterate
        b = pow(c, 2**(m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return [x, p-x]

def inv(a, n):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % n, n
    while low > 1:
        r = high//low
        nm, new = hm-lm*r, high-low*r
        lm, low, hm, high = nm, new, lm, low
    return lm % n

# Pre-compute (i) a list of primes
# (ii) a list of -1 legendre bases for each prime
# (iii) the inverse for each base
LENPRIMES = 1000
primes = []
r = 2**31 - 1
for i in range(LENPRIMES):
    r += 2
    while pow(2, r, r) != 2: r += 2
    primes.append(r)
bases = [None] * LENPRIMES
invbases = [None] * LENPRIMES
for i in range(LENPRIMES):
    b = 2
    while legendre_symbol(b, primes[i]) == 1:
        b += 1
    bases[i] = b
    invbases[i] = inv(b, primes[i])

# Compute the PoW
def forward(val, rounds=10**6):
    t1 = time.time()
    for i in range(rounds):
        # Select a prime
        p = primes[i % LENPRIMES]
        # Make sure the value we're working on is a
        # quadratic residue. If it's not, do a spooky
        # transform (ie. multiply by a known
        # non-residue) to make sure that it is
        if legendre_symbol(val, p) != 1:
            val = (val * invbases[i % LENPRIMES]) % p
            mul_by_base = 1
        else:
            mul_by_base = 0
        # Take advantage of the fact that two square
        # roots exist to hide whether or not the spooky
        # transform was done in the result so that we
        # can invert it when verifying
        val = sorted(prime_mod_sqrt(val, p))[mul_by_base]
    print time.time() - t1
    return val

def backward(val, rounds=10**6):
    t1 = time.time()
    for i in range(rounds-1, -1, -1):
        # Select a prime
        p = primes[i % LENPRIMES]
        # Extract the info about whether or not the
        # spooky transform was done
        mul_by_base = val * 2 > p
        # Square the value (ie. invert the square root)
        val = pow(val, 2, p)
        # Undo the spooky transform if needed
        if mul_by_base:
            val = (val * bases[i % LENPRIMES]) % p
    print time.time() - t1
    return val
