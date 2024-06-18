from fast_fft import (
    reverse_bit_order, log2, M31, M31SQ, HALF, invx, invy, fft,
    to_extension_field, extension_field_mul, modinv, np,
    array, zeros, tobytes, arange, folded_rbos
)
from merkle import merkelize, hash, get_branch, verify_branch

BASE_CASE_SIZE = 128
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND
NUM_CHALLENGES = 80

def chunkify(values):
    o = tobytes(values)
    return [
        o[i:i+16*FOLD_SIZE_RATIO]
        for i in range(0, len(o), 16*FOLD_SIZE_RATIO)
    ]

# Get the Merkle branch challenge indices from a root
def get_challenges(root, domain_size, num_challenges):
    challenge_data = b''.join(
        hash(root + bytes([i])) for i in range((num_challenges + 7) // 8)
    )
    return array([
        int.from_bytes(challenge_data[i:i+4], 'little') % domain_size
        for i in range(0, num_challenges * 4, 4)
    ])

def modinv_ext(x):
    x = x.swapaxes(0, x.ndim-1)
    r20 = (x[2] * x[2] + M31SQ - x[3] * x[3]) % M31
    r21 = (2 * x[2] * x[3]) % M31
    denom0 = (x[0]**2 - x[1]**2 + r20 - r21 * 2) % M31
    denom1 = (2*x[0]*x[1] + r21 + r20 * 2) % M31
    inv_denom_norm = modinv((denom0 ** 2 + denom1 ** 2) % M31)
    inv_denom0 = (denom0 * inv_denom_norm) % M31
    inv_denom1 = (M31SQ - denom1 * inv_denom_norm) % M31
    o = np.stack((
        x[0] * inv_denom0 + M31SQ - x[1] * inv_denom1,
        x[0] * inv_denom1 + x[1] * inv_denom0,
        M31SQ - x[2] * inv_denom0 + x[3] * inv_denom1,
        M31SQ * 2 - x[2] * inv_denom1 - x[3] * inv_denom0,
    ))
    return (o % M31).swapaxes(0, o.ndim-1)

def extension_point_add(pt1, pt2):
    return np.stack((
        extension_field_mul(pt1[0], pt2[0])
        + M31SQ - extension_field_mul(pt1[1], pt2[1]),
        extension_field_mul(pt1[0], pt2[1]) +
        extension_field_mul(pt1[1], pt2[0])
    )) % M31

def rbo_index_to_original(length, index):
    if length == 1:
        return np.zeros_like(index)
    sub = rbo_index_to_original(length >> 1, index >> 1)
    return (1 - (index % 2)) * sub + (index % 2) * (length - 1 - sub)

def fold(values, coeff, first_round):
    for i in range(FOLDS_PER_ROUND):
        full_len, half_len = values.shape[-2], values.shape[-2]//2
        left, right = values[::2], values[1::2]
        f0 = ((left + right) * HALF) % M31
        if i == 0 and first_round:
            twiddle = \
                invy[full_len: full_len + half_len][folded_rbos[half_len:full_len]]
        else:
            twiddle = \
                invx[full_len*2: full_len*2 + half_len][folded_rbos[half_len:full_len]]
        twiddle_box = np.zeros_like(left)
        twiddle_box[:] = twiddle.reshape((half_len,) + (1,) * (left.ndim-1))
        f1 = ((((left + M31 - right) * HALF) % M31) * twiddle_box) % M31
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
        twiddle_box = np.zeros_like(left)
        twiddle_box[:] = twiddle.reshape((left.shape[0],) + (1,)*(left.ndim-1))
        f1 = ((((left + M31 - right) * HALF) % M31) * twiddle_box) % M31
        values = (f0 + extension_field_mul(f1, coeff)) % M31
        positions = positions[::2]
        domain_size //= 2
    return values

def prove_low_degree(evaluations):
    assert len(evaluations.shape) == 2 and evaluations.shape[-1] == 4
    # Commit Merkle root
    values = evaluations[folded_rbos[len(evaluations):len(evaluations)*2]]
    leaves = []
    trees = []
    roots = []
    # Prove descent
    rounds = log2(len(evaluations) // BASE_CASE_SIZE) // FOLDS_PER_ROUND
    print("Generating FRI proof")
    for i in range(rounds):
        leaves.append(values)
        trees.append(merkelize(chunkify(values)))
        roots.append(trees[-1][1])
        print('Root: 0x{}'.format(roots[-1].hex()))
        print("Descent round {}: {} values".format(i+1, len(values)))
        fold_factor = get_challenges(b''.join(roots), M31, 4)
        print("Fold factor: {}".format(fold_factor))
        values = fold(values, fold_factor, i==0)
    entropy = b''.join(roots) + tobytes(values)
    challenges = get_challenges(
        entropy, len(evaluations) >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    round_challenges = (
        challenges.reshape((1,)+challenges.shape)
        >> arange(0, rounds * FOLDS_PER_ROUND, FOLDS_PER_ROUND)
        .reshape((rounds,) + (1,) * challenges.ndim)
    )

    branches = [
        [get_branch(tree, c) for c in r_challenges]
        for i, (r_challenges, tree) in enumerate(zip(round_challenges, trees))
    ]
    round_challenges_xfold = zeros(round_challenges.shape + (FOLD_SIZE_RATIO,))
    round_challenges_xfold = round_challenges.reshape(round_challenges.shape + (1,)) * 8 + arange(FOLD_SIZE_RATIO).reshape(1, 1, FOLD_SIZE_RATIO)

    leaf_values = [
        leaves[i][round_challenges_xfold[i]]
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
    print("Verifying FRI proof")
    # Prove descent
    entropy = b''.join(roots) + tobytes(final_values)
    challenges = get_challenges(
        entropy, len_evaluations >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    for i in range(len(roots)):
        print("Descent round {}".format(i+1))
        fold_factor = get_challenges(b''.join(roots[:i+1]), M31, 4)
        print("Fold factor: {}".format(fold_factor))
        evaluation_size = len_evaluations >> (i * FOLDS_PER_ROUND)
        positions = (
            challenges.reshape((NUM_CHALLENGES, 1)) * FOLD_SIZE_RATIO
            + arange(FOLD_SIZE_RATIO)
        ).reshape((NUM_CHALLENGES * FOLD_SIZE_RATIO))
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
                    arange(NUM_CHALLENGES),
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
    o[rbo_index_to_original(N, arange(N))] = final_values
    coeffs = fft(o, is_top_level=False)
    assert not np.any(coeffs[N//2:])
    return True
