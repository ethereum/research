from hashlib import sha256
def hash(x): return sha256(x).digest()

def merkle_tree(leaves):
    o = [0] * len(leaves) + leaves
    for i in range(len(leaves)-1, 0, -1):
        o[i] = hash(o[i*2] + o[i*2+1])
    return o

def log2(x):
    return 0 if x == 1 else 1 + log2(x//2)

def get_proof_indices(tree_indices, depth):
    leaf_count = 2**depth
    # Get all indices touched by the proof
    maximal_indices = set({})
    for i in tree_indices:
        x = leaf_count + i 
        while x > 1:
            maximal_indices.add(x ^ 1)
            x //= 2 
    maximal_indices = [leaf_count + i for i in tree_indices] + sorted(list(maximal_indices))[::-1]
    # Get indices that cannot be recalculated from earlier indices
    redundant_indices = set({})
    proof = []
    for index in maximal_indices:
        if index not in redundant_indices:
            proof.append(index)
            while index > 1:
                redundant_indices.add(index)
                if (index ^ 1) not in redundant_indices:
                    break
                index //= 2 
    return [i for i in proof if i-leaf_count not in tree_indices]

def mk_multi_proof(tree, indices):
    return [tree[i] for i in get_proof_indices(indices, log2(len(tree) // 2))]

def verify_multi_proof(root, indices, leaves, depth, proof):
    tree = {}
    for index, leaf in zip(indices, leaves):
        tree[2**depth + index] = leaf
    for index, proofitem in zip(get_proof_indices(indices, depth), proof):
        tree[index] = proofitem
    indexqueue = sorted(tree.keys())[:-1]
    i = 0
    while i < len(indexqueue):
        index = indexqueue[i]
        if index >= 2 and index^1 in tree:
            tree[index//2] = hash(tree[index - index%2] + tree[index - index%2 + 1])
            indexqueue.append(index//2)
        i += 1
    return (indices == []) or (1 in tree and tree[1] == root)
