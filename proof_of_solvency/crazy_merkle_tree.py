# A different alternative hash-based proof of solvency protocol. Basically,
# instead of building a Merkle sum tree, build a regular Merkle tree,
# where there is a separate leaf for every unit of currency (satoshi, wei, etc)
#
# It turns out that you can build such a tree and build proofs for such a tree
# very efficiently, in N * log(B) time, where N is the number of users and B is
# the total balance. NO O(B) calculation is required!
# 
# The trick is to realize that given even a billion nodes with the same contents
# that are beside each other in the tree, computing the 500m nodes above requires
# at most three hashes (one for all inner nodes, and two to compute H(A, B)
# and H(B, C) in the case where you have [... A A B B ... B C C ... ] in the tree,
# if the transitions from A to B and B to C both happen at odd indices)
#
# This approach requires a bit more hashing, but it offers somewhat stronger privacy
# compared to the Merkle sum approach because it does not reveal sister balances,
# and it has a clearer argument for why "negative balance attacks" are not an issue.
# Similar ideas were used in 2018-era Plasma Cash protocol iterations.
#
# This code only computes the Merkle root. Computing the left-side and right-side
# branches to give to the user as proof of inclusion is left as an exercise to the
# reader.

import hashlib
import copy

def hash(x):
    return hashlib.sha256(x).digest()

def is_power_of_2(x):
    return x & (x-1) == 0

def crazy_merkle(values):
    assert is_power_of_2(sum(x[1] for x in values))
    # Base case: single value, width 1
    if len(values) == 1 and values[0][1] == 1:
        return values[0][0]
    # Recursive case
    next_layer = []
    subtract_from_next = 0
    for i in range(len(values)):
        count = values[i][1] - subtract_from_next
        if count >= 2:
            next_layer.append((hash(values[i][0] * 2), count // 2))
        if count % 2 == 1:
            next_layer.append((hash(values[i][0] + values[i+1][0]), 1))
            subtract_from_next = 1
        else:
            subtract_from_next = 0
    return crazy_merkle(next_layer)

def flatten(values):
    o = []
    for value, repeats in values:
        o.extend([value] * repeats)
    return o

def basic_merkle(items):
    assert is_power_of_2(len(items))
    o = [None] * len(items) + items
    for i in range(len(items) - 1, 0, -1):
        o[i] = hash(o[i*2] + o[i*2+1])
    return o[1]

def test():
    values = [(i.to_bytes(32, 'big'), i**2) for i in range(1, 100)]
    values.append((b'doge'*8, 2**19 - sum(i**2 for i in range(1, 100))))
    x1 = crazy_merkle(values)
    x2 = basic_merkle(flatten(values))
    assert x1 == x2
    print('root match')

if __name__ == '__main__':
    test()
