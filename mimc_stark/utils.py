from merkle_tree import blake

# Get the set of powers of R, until but not including when the powers
# loop back to 1
def get_power_cycle(r, modulus):
    o = [1, r]
    while o[-1] != 1:
        o.append((o[-1] * r) % modulus)
    return o[:-1]

# Extract pseudorandom indices from entropy
def get_pseudorandom_indices(seed, modulus, count, exclude_multiples_of=0):
    assert modulus < 2**24
    data = seed
    while len(data) < 4 * count:
        data += blake(data[-32:])
    if exclude_multiples_of == 0:
        return [int.from_bytes(data[i: i+4], 'big') % modulus for i in range(0, count * 4, 4)]
    else:
        real_modulus = modulus * (exclude_multiples_of - 1) // exclude_multiples_of
        o = [int.from_bytes(data[i: i+4], 'big') % real_modulus for i in range(0, count * 4, 4)]
        return [x+1+x//(exclude_multiples_of-1) for x in o]

def is_a_power_of_2(x):
    return True if x==1 else False if x%2 else is_a_power_of_2(x//2)
