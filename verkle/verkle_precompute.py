from py_ecc import optimized_bls12_381 as b
from fft import fft
from poly_utils import PrimeField
from multicombs import lincomb
from fk20 import generate_all_proofs
import os
import hashlib
import random
import time
from verkle import (
    WIDTH,
    DEPTH,
    MODULUS,
    field,
    root_of_unity_candidates,
    ROOT_OF_UNITY,
    POWERS, 
    INVERSES,
    LAGRANGE_POLYS,
    generate_setup,
    hash,
    hash_point_to_field,
    layer_commit,
    generate_quotient,
    verify_proof
)

# Generates the data and commitent tree for a piece of data
# as well as the precomputed proofs
def generate_tree(data, setup):
    data += [0] * (WIDTH ** DEPTH - len(data))
    data_tree = [data]
    proof_tree = []
    commitment_tree = []
    for d in range(DEPTH-1, -1, -1):
        new_commitment_layer = []
        new_proof_layer = []
        new_proof_layer2 = []
        for pos in range(0, len(data_tree[0]), WIDTH):
            new_commitment_layer.append(layer_commit(data_tree[0][pos: pos+WIDTH], setup))
            # n^2 proof computation -- replaced by FK20
            #for sub_index in range(0, WIDTH):
            #     new_proof_layer.append(lincomb(setup[2], generate_quotient(data_tree[0][pos: pos+WIDTH], sub_index), b.add, b.Z1))

            new_proof_layer += generate_all_proofs(data_tree[0][pos: pos+WIDTH], setup)

        commitment_tree.insert(0, new_commitment_layer)
        proof_tree.insert(0, new_proof_layer)
        if d > 0:
            data_tree.insert(0, [hash_point_to_field(c) for c in commitment_tree[0]])
    assert len(data_tree) == len(commitment_tree) == len(proof_tree)
    assert len(commitment_tree[0]) == 1
    return data_tree, commitment_tree, proof_tree

# Generate a witness proving a particular set of indices
def generate_proof(data_tree, commitment_tree, proof_tree, indices, setup):
    committee_root = commitment_tree[0][0]
    # Generate a random r value; we use a power of r as a coefficient for each sub-leaf
    # to create a random linear combination
    r = int.from_bytes(hash(str([committee_root[0].n] + indices).encode('utf-8')), 'big') % b.curve_order
    #print("r", r)
    
    # Total polynomial that we are evaluating
    total_poly_evaluations = [0] * WIDTH
    # The set of all intermediate commitments
    commitments = []
    total_proofs = b.Z1

    for i, index in enumerate(indices):
        c = []
        # Walk from top to bottom of the tree
        for d in range(DEPTH):
            # Power of r for this leaf
            rfactor = pow(r, i*DEPTH+d, MODULUS)
            # Position of the index in this layer of data
            position_of_leaf = index // WIDTH**(DEPTH-d-1)
            # Position of the index within its data chunk
            sub_index = position_of_leaf % WIDTH
            proof = proof_tree[d][position_of_leaf]

            # Add in rfactor*D / (X - w**i) to the total
            total_proofs = b.add(total_proofs, b.multiply(proof, rfactor))
            # Provide as part of the proof all intermediate-level commitments
            if d > 0:
                c.append(commitment_tree[d][position_of_leaf // WIDTH])
        commitments.append(c)
    # Generate a polynomial commitment for the result
    return commitments, b.normalize(total_proofs)

def test():
    setup = generate_setup(1927409816240961209460912649124)
    print("Generated setup")
    data = [(1720941241 + (i**70) ^ (i**99)) % 2**200 for i in range(WIDTH ** DEPTH)]
    print("Generated random test data")
    a = time.time()
    data_tree, commitment_tree, proof_tree = generate_tree(data, setup)
    print("Generated commitment and proofs in %.3f seconds" % (time.time() - a))
    print("Generated data and commitment tree")
    print("Root: ", commitment_tree[0][0])
    print('-------------------')
    coords = [729 % WIDTH ** DEPTH, 505 % WIDTH ** DEPTH]
    a = time.time()
    commitments, w = generate_proof(data_tree, commitment_tree, proof_tree, coords, setup)
    print("Generated proof in %.3f seconds" % (time.time() - a))
    print('-------------------')
    print("Witness: ", commitments, w)
    a = time.time()
    assert verify_proof((commitments, w), commitment_tree[0][0], coords, [data[c] for c in coords], setup)
    print("Verified proof in %.3f seconds" % (time.time() - a))

if __name__ == '__main__':
    test()
