from hashlib import blake2s

def hash(x): return blake2s(x).digest()[:32]

def numhash(x, i, seed, modulus):
    assert 0 <= i < 4
    return (int.from_bytes(hash(x.to_bytes(32, 'big') + seed), 'big') // modulus**i) % modulus

def numhash_all(x, seed, modulus):
    h = int.from_bytes(hash(x.to_bytes(32, 'big') + seed), 'big')
    return [(h // modulus ** i) % modulus for i in range(4)]

def next_perfect_square(n):
    if int(n ** 0.5) ** 2 == n:
        return n
    return (int(n ** 0.5) + 1) ** 2

def multi_feistel(modulus, xs, seed, precompute=False):
    h = int(next_perfect_square(modulus) ** 0.5)

    numhashes = [numhash_all(i, seed, modulus) for i in range(h)] if precompute else None

    o = []
    for x in xs:
        while 1:
            L, R = x//h, x%h
            for i in range(4):
                if precompute:
                    new_R = (L + numhashes[R][i]) % h
                else:
                    new_R = (L + numhash(R, i, seed, modulus)) % h
                L = R
                R = new_R
            x = L * h + R
            if x < modulus:
                o.append(x)
                break
    return o

def feistel_shuffle(values, seed):
    return [values[i] for i in multi_feistel(len(values), list(range(len(values))), seed, True)]

def feistel_shuffle_partial(values, seed, count):
    return [values[i] for i in multi_feistel(len(values), list(range(count)), seed, False)]
