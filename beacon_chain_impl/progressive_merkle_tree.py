from hashlib import blake2s
import binascii

def hash(x): return blake2s(x).digest()

zerohashes = [b'\x00' * 32]
for i in range(1, 32):
    zerohashes.append(hash(zerohashes[i-1] + zerohashes[i-1]))

# Add a value to a Merkle tree by using the algo
# that stores a branch of sub-roots
def add_value(branch, index, value):
    i = 0
    while (index+1) % 2**(i+1) == 0:
        i += 1
    for j in range(0, i):
        value = hash(branch[j] + value)
        # branch[j] = zerohashes[j]
    branch[i] = value

# Compute a Merkle root the dumb way
def merkle_root(values):
    for h in range(32):
        if len(values) % 2 == 1:
            values.append(zerohashes[h])
        values = [hash(values[i] + values[i+1]) for i in range(0, len(values), 2)]
    return values[0]

def get_root_from_branch(branch, size):
    r = b'\x00' * 32
    for h in range(32):
        if (size >> h) % 2 == 1:
            r = hash(branch[h] + r)
        else:
            r = hash(r + zerohashes[h])
    return r

def branch_by_branch(values):
    branch = zerohashes[::]
    # Construct the tree using the branch-based algo
    for index, value in enumerate(values):
        add_value(branch, index, value)
    # Return the root
    return get_root_from_branch(branch, len(values))

testdata = [(i + 2**255).to_bytes(32, 'big') for i in range(10000)]

# The Merkle root algo assumes trailing zero bytes
assert merkle_root(testdata[:5]) == merkle_root(testdata[:5] + [b'\x00' * 32] * 5)

# Verify equivalence of the simple all-at-once method and the progressive method
assert branch_by_branch(testdata[:1]) == merkle_root(testdata[:1])
assert branch_by_branch(testdata[:2]) == merkle_root(testdata[:2])
assert branch_by_branch(testdata[:3]) == merkle_root(testdata[:3])
assert branch_by_branch(testdata[:5049]) == merkle_root(testdata[:5049])
