try:
    import cupy as np
except:
    import numpy as np

from fast_fft import (
    reverse_bit_order, log2, M31, M31SQ, HALF, invx, invy
)
from merkle import merkelize, hash, get_branch, verify_branch

BASE_CASE_SIZE = 128
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND
NUM_CHALLENGES = 80
EXTENSION_I = 2

def chunkify(values):
    o = values.astype(np.uint32).tobytes()
    return [
        o[i:i+16*FOLD_SIZE_RATIO]
        for i in range(0, len(o), 16*FOLD_SIZE_RATIO)
    ]

def folded_reverse_bit_order(vals):
    vals = np.array(vals, dtype=np.uint64)
    size = vals.shape[0]
    shape_suffix = vals.shape[1:]
    for i in range(log2(size)):
        vals = np.reshape(vals, (1 << i, size >> i) + shape_suffix)
        full_len = vals.shape[1]
        half_len = full_len >> 1
        vals[:, half_len:] = vals[:, full_len-1: half_len-1: -1]
    return reverse_bit_order(vals.reshape((size,) + shape_suffix))

# Get the Merkle branch challenge indices from a root
def get_challenges(root, domain_size, num_challenges):
    return np.array([
        int.from_bytes(hash(root+bytes([i])), 'little') % domain_size
        for i in range(num_challenges)
    ], dtype=np.uint64)

"""
         o_LL = [m1*o1 - m2*o2, m1*o2 + m2*o1]
        o_LR = [m1*o3 - m2*o4, m1*o4 + m2*o3]
        o_RL = [m3*o1 - m4*o2, m3*o2 + m4*o1]
        o_RR = [m3*o3 - m4*o4, m3*o4 + m4*o3]
        o = [
            o_LL[0] - (o_RR[0] - o_RR[1]*self.extension_i),
            o_LL[1] - (o_RR[1] + o_RR[0]*self.extension_i),
            o_LR[0] + o_RL[0],
            o_LR[1] + o_RL[1]
        ]
"""

def extension_field_mul(A, B):
    # todo: needs moar karatsuba
    A = A.transpose()
    B = B.transpose()
    o_LL = [A[0] * B[0] + M31SQ - A[1] * B[1], A[0] * B[1] + A[1] * B[0]]
    o_LR = [A[0] * B[2] + M31SQ - A[1] * B[3], A[0] * B[3] + A[1] * B[2]]
    o_RL = [A[2] * B[0] + M31SQ - A[3] * B[1], A[2] * B[1] + A[3] * B[0]]
    o_RR = [A[2] * B[2] + M31SQ - A[3] * B[3], A[2] * B[3] + A[3] * B[2]]
    o = np.array([
        o_LL[0] + M31SQ - o_RR[0] + (o_RR[1] % M31) * EXTENSION_I,
        o_LL[1] + M31SQ - o_RR[1] - (o_RR[0] % M31) * EXTENSION_I,
        o_LR[0] + o_RL[0],
        o_LR[1] + o_RL[1]
    ])
    return (o % M31).transpose()

def fold(values, coeff, first_round):
    for i in range(FOLDS_PER_ROUND):
        full_len, half_len = values.shape[-2], values.shape[-2]//2
        left, right = values[::2], values[1::2]
        f0 = ((left + right) * HALF) % M31
        if i == 0 and first_round:
            twiddle = folded_reverse_bit_order(
                invy[full_len: full_len + half_len]
            )
        else:
            twiddle = folded_reverse_bit_order(
                invx[full_len*2: full_len*2 + half_len]
            )
        twiddle = np.expand_dims(twiddle, axis=tuple(range(1, len(f0.shape))))
        f1 = ((((left + M31 - right) * HALF) % M31) * twiddle) % M31
        values = (f0 + extension_field_mul(f1, coeff)) % M31
    return values

def to_extension_field(values):
    return np.pad(values[...,np.newaxis], ((0,0), (0,3)))

def prove_low_degree(evaluations):
    assert len(evaluations.shape) == 2 and evaluations.shape[-1] == 4
    # Commit Merkle root
    values = folded_reverse_bit_order(evaluations)
    leaves = []
    trees = []
    roots = []
    # Prove descent
    rounds = log2(len(evaluations) // BASE_CASE_SIZE) // FOLDS_PER_ROUND
    print("Generating proof")
    for i in range(rounds):
        leaves.append(values)
        trees.append(merkelize(chunkify(values)))
        roots.append(trees[-1][1])
        print('Root:', roots[-1])
        print("Descent round {}: {} values".format(i+1, len(values)))
        fold_factor = get_challenges(b''.join(roots), M31, 4)
        print("Fold factor: {}".format(fold_factor))
        values = fold(values, fold_factor, i==0)
    entropy = b''.join(roots + [x.astype(np.uint32).tobytes() for x in values])
    challenges = get_challenges(
        entropy, len(evaluations) >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    round_challenges = [
        [c >> (i * FOLDS_PER_ROUND) for c in challenges]
        for i in range(rounds+1)
    ]
    branches = [
        [get_branch(tree, c) for c in r_challenges]
        for i, (r_challenges, tree) in enumerate(zip(round_challenges, trees))
    ]
    leaf_values = [
        [
            leaf_list[c * FOLD_SIZE_RATIO: (c+1) * FOLD_SIZE_RATIO]
            for c in r_challenges
        ]
        for (leaf_list, r_challenges) in zip(leaves, round_challenges)
    ]
    return {
        "roots": roots,
        "branches": branches,
        "leaf_values": leaf_values,
        "final_values": values
    }
