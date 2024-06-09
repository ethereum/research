from hashlib import sha256

def hash(x):
    return sha256(x).digest()

# Build a Merkle tree from the inputs, where o[i] is the parent node of
# o[2i] and o[2i+1], the second half of o is the original data, and o[1]
# is the root
def merkelize(vals):
    assert len(vals) & (len(vals)-1) == 0
    o = [None] * len(vals) + [hash(x) for x in vals]
    for i in range(len(vals)-1, 0, -1):
        o[i] = hash(o[i*2] + o[i*2+1])
    return o

def get_root(tree):
    return tree[1]

# Get a Merkle branch for the value at a particular position
def get_branch(tree, pos):
    offset_pos = int(pos + len(tree) // 2)
    branch_length = len(tree).bit_length() - 2
    return [tree[(offset_pos >> i)^1] for i in range(branch_length)]

# Verify that Merkle branch (requires only the root, not the tree)
def verify_branch(root, pos, val, branch):
    x = hash(val)
    for b in branch:
        if pos & 1:
            x = hash(b + x)
        else:
            x = hash(x + b)
        pos //= 2
    return x == root

import concurrent.futures

def merkelize_by_levels(vals):
    assert len(vals) & (len(vals)-1) == 0
    levels = [[hash(x) for x in vals]]
    while len(levels[-1]) > 1:
        L = levels[-1]
        levels.append([hash(L[i] + L[i+1]) for i in range(0, len(L), 2)])
    return levels[::-1]

# Function to execute in parallel
def pooled_merkelize(data):
    inputs = [
        data[i * len(data) // 8: (i+1) * len(data) // 8]
        for i in range(8)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # Submit the function with each input to the executor
        futures = [executor.submit(merkelize_by_levels, i) for i in inputs]
        
        # Collect the results as they complete
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        head = merkelize([x[0][0] for x in results])[:8]
        output = head
        for i in range(len(results[0])):
            for result in results:
                output.extend(result[i])
    
    return output
