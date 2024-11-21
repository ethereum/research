from py_ecc import optimized_bls12_381 as b
from fft import fft
from poly_utils import PrimeField
from multicombs import lincomb
import os
import hashlib
import random
import time

WIDTH = 16
DEPTH = 3

MODULUS = b.curve_order

field = PrimeField(MODULUS)

# Root of unity for a given evaluation domain size
root_of_unity_candidates = {
    512: 12531186154666751577774347439625638674013361494693625348921624593362229945844,
    256: 21071158244812412064791010377580296085971058123779034548857891862303448703672,
    128: 3535074550574477753284711575859241084625659976293648650204577841347885064712,
    64: 6460039226971164073848821215333189185736442942708452192605981749202491651199,
    32: 32311457133713125762627935188100354218453688428796477340173861531654182464166,
    16: 35811073542294463015946892559272836998938171743018714161809767624935956676211
}

ROOT_OF_UNITY = root_of_unity_candidates[WIDTH]

assert pow(ROOT_OF_UNITY, WIDTH // 2, MODULUS) != 1
assert pow(ROOT_OF_UNITY, WIDTH, MODULUS) == 1

# Powers of the root of unity
POWERS = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(WIDTH)]

# 1/(x-r) evaluated at every r, for every x in the set of powers
INVERSES = [field.multi_inv([field.sub(p, x) for p in POWERS]) for x in POWERS]

# Polynomials that evaluate to [000....010....000] across the evaluation domain,
# one for every possible position of the 1
LAGRANGE_POLYS = [
    fft([0]*i + [1] + [0]*(WIDTH-1-i), MODULUS, ROOT_OF_UNITY, inv=True)
    for i in range(WIDTH)
]

# Generate trusted setup, both in coefficient and Lagrange form
def generate_setup(s):
    return (
        [b.multiply(b.G1, pow(s, i, MODULUS)) for i in range(WIDTH+1)],
        [b.multiply(b.G2, pow(s, i, MODULUS)) for i in range(WIDTH+1)],
        [b.multiply(b.G1, field.eval_poly_at(l, s)) for l in LAGRANGE_POLYS],
        [b.multiply(b.G2, field.eval_poly_at(l, s)) for l in LAGRANGE_POLYS],
    )

# Just sha256
def hash(x):
    return hashlib.sha256(x).digest()

# Hashes a curve point to a number
def hash_point_to_field(pt):
    assert len(pt) == 2
    return int.from_bytes(hash(str(pt).encode('utf-8')), 'big') % MODULUS

# Generates a commitment to a layer
def layer_commit(values, setup):
    values += [0] * (WIDTH - len(values))
    coeffs = fft(values, MODULUS, ROOT_OF_UNITY, inv=True)
    return b.normalize(lincomb(setup[0][:len(coeffs)], coeffs, b.add, b.Z1))

# Generates the polynomial D / (x-w**i) in evaluation form, for a given data and a given index i
def generate_quotient(values, index):
    x = pow(ROOT_OF_UNITY, index, MODULUS)
    P = [field.sub(v, values[index]) for v in values]
    P[index] = 0
    inv_Q = INVERSES[index]
    P_over_Q = [field.mul(a,b) for a,b in zip(P, inv_Q)]
    top_coeff = field.div(sum([field.mul(a, p) for a, p in zip(P_over_Q, POWERS)]), WIDTH)
    lagrange_coefficient = field.div(top_coeff, LAGRANGE_POLYS[index][-1])
    P_over_Q[index] = MODULUS - lagrange_coefficient
    #P_over_Q_coeffs = fft(P_over_Q, MODULUS, ROOT_OF_UNITY, inv=True)
    #assert P_over_Q_coeffs[-1] == 0
    #assert fft(field.mul_polys(P_over_Q_coeffs[:-1], [-x, 1]), MODULUS, ROOT_OF_UNITY) == P
    return P_over_Q

# Generates the data and commitent tree for a piece of data
# The i'th commitment layer is the set of commitments of the i'th data layer
# so len(commitment_tree[i]) == len(data_tree[i]) * WIDTH
def generate_tree(data, setup):
    data += [0] * (WIDTH ** DEPTH - len(data))
    data_tree = [data]
    commitment_tree = []
    for d in range(DEPTH-1, -1, -1):
        new_commitment_layer = []
        for pos in range(0, len(data_tree[0]), WIDTH):
            new_commitment_layer.append(layer_commit(data_tree[0][pos: pos+WIDTH], setup))
        commitment_tree.insert(0, new_commitment_layer)
        if d > 0:
            data_tree.insert(0, [hash_point_to_field(c) for c in commitment_tree[0]])
    assert len(data_tree) == len(commitment_tree)
    assert len(commitment_tree[0]) == 1
    return data_tree, commitment_tree

# Generate a witness proving a particular set of indices
def generate_proof(data_tree, commitment_tree, indices, setup):
    committee_root = commitment_tree[0][0]
    # Generate a random r value; we use a power of r as a coefficient for each sub-leaf
    # to create a random linear combination
    r = int.from_bytes(hash(str([committee_root[0].n] + indices).encode('utf-8')), 'big') % b.curve_order
    #print("r", r)
    
    # Total polynomial that we are evaluating
    total_poly_evaluations = [0] * WIDTH
    # The set of all intermediate commitments
    commitments = []

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
            #print('d', d, 'i', index, 'rfactor', rfactor, 'pos', position_of_leaf)
            data = data_tree[d][position_of_leaf - sub_index: position_of_leaf - sub_index + WIDTH]
            quotient = generate_quotient(data, sub_index)
            # Add in rfactor*D / (X - w**i) to the total
            total_poly_evaluations = [a+b*rfactor for a,b in zip(total_poly_evaluations, quotient)]
            # Provide as part of the proof all intermediate-level commitments
            if d > 0:
                c.append(commitment_tree[d][position_of_leaf // WIDTH])
        commitments.append(c)
    # Generate a polynomial commitment for the result
    return commitments, b.normalize(lincomb(setup[2], total_poly_evaluations, b.add, b.Z1))

# Verify a witness generated by the above function
def verify_proof(proof, commitment_root, indices, values, setup):
    # Regenerate the same r value as above
    r = int.from_bytes(hash(str([commitment_root[0].n] + indices).encode('utf-8')), 'big') % b.curve_order
    #print("r", r)
    commitments, witness = proof
    # We're making a big pairing check that essentially checks the equation:
    # sum [(P_i - y_i) * r_i * Z(everything except x_i)] = w * Z(everything) = sum [Q_i * r_i * Z(everything)]
    # where Z(set) = product: (X - s) for s in set
    pairing_check = b.FQ12.one()
    for i, (c, index, v) in enumerate(zip(commitments, indices, values)):
        for d in range(DEPTH):
            rfactor = pow(r, i*DEPTH+d, MODULUS)
            position_of_leaf = index // WIDTH**(DEPTH-d-1)
            # Position of this leaf in the data
            sub_index = position_of_leaf % WIDTH
            #print('d', d, 'i', index, 'rfactor', rfactor, 'pos', position_of_leaf)
            # P_i
            comm = c[d-1] if d else commitment_root
            comm = (comm[0], comm[1], b.FQ.one())
            leaf = hash_point_to_field(c[d]) if d < DEPTH-1 else v
            #print('comm', comm, 'subindex', sub_index, 'leaf', leaf)
            # (P_i - y_i) * r_i
            comm_minus_leaf_times_r = b.multiply(b.add(comm, b.multiply(b.G1, MODULUS - leaf)), rfactor)
            # Z(everything except x_i)
            Z_comm = b.multiply(setup[3][sub_index], field.inv(LAGRANGE_POLYS[sub_index][-1]))
            # Add the product into the pairing
            pairing_check *= b.pairing(Z_comm, comm_minus_leaf_times_r, False)
    # Z(everything)
    global_Z_comm = b.add(setup[1][WIDTH], b.neg(setup[1][0]))
    # Subtract out sum [Q_i * r_i * Z(everything)]
    pairing_check *= b.pairing(b.neg(global_Z_comm), (witness[0], witness[1], b.FQ.one()), False)
    o = b.final_exponentiate(pairing_check)
    assert o == b.FQ12.one(), o
    return o == b.FQ12.one()

def test():
    setup = generate_setup(1927409816240961209460912649124)
    print("Generated setup")
    data = [(1720941241 + (i**70) ^ (i**99)) % 2**200 for i in range(WIDTH ** DEPTH)]
    print("Generated random test data")
    data_tree, commitment_tree = generate_tree(data, setup)
    print("Generated data and commitment tree")
    print("Root: ", commitment_tree[0][0])
    coords = [729 % WIDTH ** DEPTH, 505 % WIDTH ** DEPTH]
    a = time.time()
    commitments, w = generate_proof(data_tree, commitment_tree, coords, setup)
    print("Generated proof in %.3f seconds" % (time.time() - a))
    print('-------------------')
    print("Witness: ", commitments, w)
    a = time.time()
    assert verify_proof((commitments, w), commitment_tree[0][0], coords, [data[c] for c in coords], setup)
    print("Verified proof in %.3f seconds" % (time.time() - a))

if __name__ == '__main__':
    test()
