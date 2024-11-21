from hashlib import blake2s

def hash(x): return blake2s(x).digest()[:32]


def values_at_position(n, positions, seed, precompute=False):
    values = positions[::]
    for round in range(32):
        if precompute:
            hashvalues = b''.join([
                hash(seed + round.to_bytes(1, 'big') + i.to_bytes(4, 'big'))
                for i in range((n + 255) // 256)
            ])
        pivot = int.from_bytes(hash(seed + round.to_bytes(1, 'big')), 'big') % n
        if precompute:
            def permute(pos):
                flip = (pivot - pos) % n
                maxpos = max(pos, flip)
                bit = (hashvalues[maxpos // 8] >> (maxpos % 8)) % 2
                return flip if bit else pos
        else:
            def permute(pos):
                flip = (pivot - pos) % n
                maxpos = max(pos, flip)
                h = hash(seed + round.to_bytes(1, 'big') + (maxpos // 256).to_bytes(4, 'big'))
                byte = h[(maxpos % 256) // 8] 
                bit = (byte >> (maxpos % 8)) % 2
                return flip if bit else pos

        values = [permute(v) for v in values]
    return values

def swap_or_not_shuffle(values, seed):
    return [values[i] for i in values_at_position(len(values), list(range(len(values))), seed, True)]

def swap_or_not_shuffle_partial(values, seed, count):
    return [values[i] for i in values_at_position(len(values), list(range(count)), seed, False)]
