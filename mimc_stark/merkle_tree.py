try:
    from hashlib import blake2s
except:
    from pyblake2 import blake2s
blake = lambda x: blake2s(x).digest()

def merkelize(L):
    # L = permute4(L)
    nodes = [b''] * len(L) + [x.to_bytes(32, 'big') if isinstance(x, int) else x for x in L]
    for i in range(len(L) - 1, 0, -1):
        nodes[i] = blake(nodes[i*2] + nodes[i*2+1])
    return nodes

def mk_branch(tree, index):
    # index = get_index_in_permuted(index, len(tree) // 2)
    index += len(tree) // 2
    o = [tree[index]]
    while index > 1:
        o.append(tree[index ^ 1])
        index //= 2
    return o

def verify_branch(root, index, proof, output_as_int=False):
    # index = get_index_in_permuted(index, 2**len(proof) // 2)
    index += 2**len(proof)
    v = proof[0]
    for p in proof[1:]:
        if index % 2:
            v = blake(p + v)
        else:
            v = blake(v + p)
        index //= 2
    assert v == root
    return int.from_bytes(proof[0], 'big') if output_as_int else proof[0]

def get_proof_indices(tree_indices, depth):
    leaf_count = 2**depth
    # Get indices included in any proof
    initial_indices = set({})
    for i in tree_indices:
        x = leaf_count + i
        while x > 1:
            initial_indices.add(x ^ 1)
            x //= 2
    initial_indices = [leaf_count + i for i in tree_indices] + sorted(list(initial_indices))[::-1]
    print('initial', initial_indices)
    # Get indices that can be removed
    redundant_indices = set({})
    indices_to_remove = set({})
    for index in initial_indices:
        print('redundant', redundant_indices, 'removed', indices_to_remove, 'checking', index)
        if index in redundant_indices:
            indices_to_remove.add(index)
        else:
            while index > 1:
                redundant_indices.add(index)
                if (index ^ 1) not in redundant_indices:
                    break
                index //= 2
    return [i for i in initial_indices if i not in indices_to_remove and i-leaf_count not in tree_indices]

def mk_multi_proof(tree, indices):
    return [tree[i] for i in get_proof_indices(indices, len(tree) // 2)]

def verify_multi_proof(root, indices, leaves, depth, proof):
    tree = {}
    for index, leaf in zip(indices, leaves):
        tree[2**depth + index] = leaf
    for index, proofitem in zip(get_proof_indices(indices, 2**depth), proof):
        tree[index] = proofitem
    indexqueue = sorted(tree.keys())[:-1]
    i = 0
    while i < len(indexqueue):
        index = indexqueue[i]
        if index^1 in tree:
            tree[index//2] = blake(tree[index - index%2] + tree[index - index%2 + 1])
            indexqueue.append(index//2)
    return 1 in tree and tree[1] == root

# Make a compressed proof for multiple indices
def mk_multi_branch(tree, indices):
    # Branches we are outputting
    output = []
    # Elements in the tree we can get from the branches themselves
    calculable_indices = {}
    for i in indices:
        new_branch = mk_branch(tree, i)
        index = len(tree) // 2 + i
        calculable_indices[index] = True
        for j in range(1, len(new_branch)):
            calculable_indices[index ^ 1] = True
            index //= 2
        output.append(new_branch)
    # Fill in the calculable list: if we can get or calculate both children, we can calculate the parent
    complete = False
    while not complete:
        complete = True
        keys = sorted([x//2 for x in calculable_indices])[::-1]
        for k in keys:
            if k*2 in calculable_indices and (k*2)+1 in calculable_indices and k not in calculable_indices:
                calculable_indices[k] = True
                complete = False
    # If for any branch node both children are calculable, or the node overlaps with a leaf, or the node
    # overlaps with a previously calculated one, elide it
    scanned = {}
    for i, b in zip(indices, output):
        index = len(tree) // 2 + i
        scanned[index] = True
        for j in range(1, len(b)):
            if ((index^1)*2 in calculable_indices and (index^1)*2+1 in calculable_indices) or ((index^1)-len(tree)//2 in indices) or (index^1) in scanned:
                b[j] = b''
            scanned[index^1] = True
            index //= 2
    return output        
    

# Verify a compressed proof
def verify_multi_branch(root, indices, proof):
    # The values in the Merkle tree we can fill in
    partial_tree = {}
    # Fill in elements from the branches
    for i, b in zip(indices, proof):
        half_tree_size = 2**(len(b) - 1)
        index = half_tree_size+i
        partial_tree[index] = b[0]
        for j in range(1, len(b)):
            if b[j]:
                partial_tree[index ^ 1] = b[j]
            index //= 2
    # If we can calculate or get both children, we can calculate the parent
    complete = False
    while not complete:
        complete = True
        keys = sorted([x//2 for x in partial_tree])[::-1]
        for k in keys:
            if k*2 in partial_tree and k*2+1 in partial_tree and k not in partial_tree:
                partial_tree[k] = blake(partial_tree[k*2] + partial_tree[k*2+1])
                complete = False
    # If any branch node is missing, we can calculate it
    for i, b in zip(indices, proof):
        index = half_tree_size + i
        for j in range(1, len(b)):
            if b[j] == b'':
                b[j] = partial_tree[index ^ 1]
            partial_tree[index^1] = b[j]
            index //= 2
    # Verify the branches and output the values
    return [verify_branch(root, i, b) for i,b in zip(indices, proof)]

# Byte length of a multi proof
def bin_length(proof):
    return sum([len(b''.join(x)) + len(x) // 8 for x in proof]) + len(proof) * 2
