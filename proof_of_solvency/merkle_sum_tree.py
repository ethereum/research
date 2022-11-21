# Example code for building and getting proofs in a Merkle sum tree,
# used to make proofs of solvency in an exchange
#
# THIS IS EDUCATIONAL CODE, NOT PRODUCTION! HIRE A SECURITY AUDITOR
# WHEN BUILDING SOMETHING FOR PRODUCTION USE.

import hashlib

# Mathematical helper methods

def hash(x):
    return hashlib.sha256(x).digest()

def log2(x):
    return len(bin(x)) - 2

def get_next_power_of_2(x):
    return 2 * get_next_power_of_2((x+1)//2) if x > 1 else 1

# Each user has a username and balance, and gets a salt generated
# This gets converted into a leaf, which does not reveal the username
def userdata_to_leaf(username, salt, balance):
    return (hash(salt + username), balance)

EMPTY_LEAF = (b'\x00' * 32, 0)

# The function for computing a parent node given two child nodes
def combine_tree_nodes(L, R):
    L_hash, L_balance = L
    R_hash, R_balance = R
    assert L_balance >= 0 and R_balance >= 0
    new_node_hash = hash(
        L_hash + L_balance.to_bytes(32, 'big') +
        R_hash + R_balance.to_bytes(32, 'big')
    )
    return (new_node_hash, L_balance + R_balance)

# Builds a full Merkle tree. Stored in flattened form where
# node i is the parent of nodes 2i and 2i+1
def build_merkle_sum_tree(user_table: "List[(username, salt, balance)]"):
    tree_size = get_next_power_of_2(len(user_table))
    tree = (
        [None] * tree_size +
        [userdata_to_leaf(*user) for user in user_table] +
        [EMPTY_LEAF for _ in range(tree_size - len(user_table))]
    )
    for i in range(tree_size - 1, 0, -1):
        tree[i] = combine_tree_nodes(tree[i*2], tree[i*2+1])
    return tree

# Root of a tree is stored at index 1 in the flattened form
def get_root(tree):
    return tree[1]

# Gets a proof for a node at a particular index
def get_proof(tree, index):
    branch_length = log2(len(tree)) - 1
    # ^ = bitwise xor, x ^ 1 = sister node of x
    index_in_tree = index + len(tree) // 2
    return [tree[(index_in_tree // 2**i) ^ 1] for i in range(branch_length)]

# Verifies a proof (duh)
def verify_proof(username, salt, balance, index, user_table_size, root, proof):
    leaf = userdata_to_leaf(username, salt, balance)
    branch_length = log2(get_next_power_of_2(user_table_size)) - 1
    for i in range(branch_length):
        if index & (2**i):
            leaf = combine_tree_nodes(proof[i], leaf)
        else:
            leaf = combine_tree_nodes(leaf, proof[i])
    return leaf == root

def test():
    import os
    user_table = [
        (b'Alice',   os.urandom(32), 20),
        (b'Bob',     os.urandom(32), 50),
        (b'Charlie', os.urandom(32), 10),
        (b'David',   os.urandom(32), 164),
        (b'Eve',     os.urandom(32), 870),
        (b'Fred',    os.urandom(32), 6),
        (b'Greta',   os.urandom(32), 270),
        (b'Henry',   os.urandom(32), 90),
    ]
    tree = build_merkle_sum_tree(user_table)
    root = get_root(tree)
    print("Root:", root)
    proof = get_proof(tree, 2)
    print("Proof:", proof)
    assert verify_proof(b'Charlie', user_table[2][1], 10, 2, 8, root, proof)
    print("Proof checked")

if __name__ == '__main__':
    test()
