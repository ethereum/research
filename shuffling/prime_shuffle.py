from hashlib import blake2s

def hash(x): return blake2s(x).digest()[:32]

def is_prime(x):
    return [i for i in range(2, int(x**0.5)+1) if x%i == 0] == []

def values_at_position(n, positions, seed, precompute=False):
    # We do the shuffling mod p, the lowest prime >= n, but if we actually shuffle into
    # the "forbidden" [n...p-1] slice we just reshuffle until we get out of that slice
    p = n 
    while not is_prime(p):
        p += 1 
    # x -> x**power is a permutation mod p
    power = 3 
    while (p-1) % power == 0 or not is_prime(power):
        power += 2 
    values = positions[::]
    power_of = [pow(i, power, p) for i in range(p)] if precompute else None
    indices = list(range(len(values)))
    for round in range(40):
        a = int.from_bytes(seed[(round % 8)*4: (round % 8)*4 + 4], 'big')
        if precompute:
            values = [(power_of[v] + a) % p for v in values]
        else:
            values = [(pow(v, power, p) + a) % p for v in values]
        for i in [i for i in indices if values[i] >= n]:
                while values[i] >= n:
                    if precompute:
                        values[i] = (power_of[values[i]] + a) % p
                    else:
                        values[i] = (pow(values[i], power, p) + a) % p
        # Update the seed if needed
        if round % 8 == 0:
            seed = hash(seed)
    return values

def prime_shuffle(values, seed):
    return [values[i] for i in values_at_position(len(values), list(range(len(values))), seed, True)]

def prime_shuffle_partial(values, seed, count):
    return [values[i] for i in values_at_position(len(values), list(range(count)), seed, False)]
