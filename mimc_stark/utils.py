from merkle_tree import blake

# Get the set of powers of R, until but not including when the powers
# loop back to 1
def get_power_cycle(r, modulus):
    o = [1, r]
    while o[-1] != 1:
        o.append((o[-1] * r) % modulus)
    return o[:-1]

# Extract pseudorandom indices from entropy
def get_pseudorandom_indices(seed, modulus, count):
    assert modulus < 2**24
    data = seed
    while len(data) < 4 * count:
        data += blake(data[-32:])
    return [int.from_bytes(data[i: i+4], 'big') % modulus for i in range(0, count * 4, 4)]
