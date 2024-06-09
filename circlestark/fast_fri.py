try:
    import cupy as np
except:
    import numpy as np

from fast_fft import (
    reverse_bit_order, log2, M31, M31SQ, HALF, invx, invy, fft
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

def rbo_index_to_original(length, index):
    if length == 1:
        return np.zeros_like(index)
    sub = rbo_index_to_original(length // 2, index // 2)
    return (1 - (index % 2)) * sub + (index % 2) * (length - 1 - sub)

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

def fold_with_positions(values, domain_size, positions, coeff, first_round):
    positions = positions[::2]
    for i in range(FOLDS_PER_ROUND):
        left, right = values[::2], values[1::2]
        f0 = ((left + right) * HALF) % M31
        if i == 0 and first_round:
            unrbo_positions = rbo_index_to_original(domain_size, positions)
            twiddle = invy[domain_size + unrbo_positions]
        else:
            unrbo_positions = rbo_index_to_original(
                domain_size * 2,
                (positions << 1) >> i
            )
            twiddle = invx[domain_size * 2 + unrbo_positions]
        twiddle = np.expand_dims(twiddle, axis=tuple(range(1, len(f0.shape))))
        f1 = ((((left + M31 - right) * HALF) % M31) * twiddle) % M31
        values = (f0 + extension_field_mul(f1, coeff)) % M31
        positions = positions[::2]
        domain_size //= 2
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
        print('Root: 0x{}'.format(roots[-1].hex()))
        print("Descent round {}: {} values".format(i+1, len(values)))
        fold_factor = get_challenges(b''.join(roots), M31, 4)
        print("Fold factor: {}".format(fold_factor))
        values = fold(values, fold_factor, i==0)
    entropy = b''.join(roots + [x.astype(np.uint32).tobytes() for x in values])
    challenges = get_challenges(
        entropy, len(evaluations) >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    round_challenges = np.broadcast_to(
        challenges[np.newaxis,:],
        (rounds, challenges.size)
    ) >> (np.arange(rounds, dtype=np.uint64) * FOLDS_PER_ROUND)[:,np.newaxis]

    branches = [
        [get_branch(tree, c) for c in r_challenges]
        for i, (r_challenges, tree) in enumerate(zip(round_challenges, trees))
    ]
    leaf_values = [
        leaves[i][
            np.broadcast_to(
                round_challenges[i][...,np.newaxis],
                round_challenges[i].shape + (FOLD_SIZE_RATIO,)
            ) * FOLD_SIZE_RATIO + np.arange(FOLD_SIZE_RATIO, dtype=np.uint64)
        ]
        for i in range(rounds)
    ]
    return {
        "roots": roots,
        "branches": branches,
        "leaf_values": leaf_values,
        "final_values": values
    }

def verify_low_degree(proof):
    roots = proof["roots"]
    branches = proof["branches"]
    leaf_values = proof["leaf_values"]
    final_values = proof["final_values"]
    len_evaluations = final_values.shape[0] << (FOLDS_PER_ROUND * len(roots))
    print("Verifying proof")
    # Prove descent
    entropy = b''.join(
        roots
        + [x.astype(np.uint32).tobytes() for x in final_values]
    )
    challenges = get_challenges(
        entropy, len_evaluations >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    for i in range(len(roots)):
        print("Descent round {}".format(i+1))
        fold_factor = get_challenges(b''.join(roots[:i+1]), M31, 4)
        print("Fold factor: {}".format(fold_factor))
        evaluation_size = len_evaluations >> (i * FOLDS_PER_ROUND)
        positions = (
            np.repeat(challenges, FOLD_SIZE_RATIO) * FOLD_SIZE_RATIO
            + np.tile(np.arange(FOLD_SIZE_RATIO, dtype=np.uint64), challenges.size)
        )
        folded_values = fold_with_positions(
            leaf_values[i].reshape((-1,4)),
            evaluation_size,
            positions,
            fold_factor,
            i==0
        )
        if i < len(roots) - 1:
            expected_values = (
                leaf_values[i+1][
                    np.arange(challenges.size),
                    challenges % FOLD_SIZE_RATIO
                ]
            )
        else:
            expected_values = final_values[challenges]
        assert np.array_equal(folded_values, expected_values)
        for j, c in enumerate(np.copy(challenges)):
            assert verify_branch(
                roots[i], c, chunkify(leaf_values[i][j])[0], branches[i][j]
            )
        challenges >>= FOLDS_PER_ROUND
    o = np.zeros_like(final_values)
    N = final_values.shape[0]
    o[rbo_index_to_original(N, np.arange(N))] = final_values
    coeffs = fft(o, is_top_level=False)
    assert np.array_equal(coeffs[N//2:], np.zeros_like(coeffs[N//2:]))
    return True
