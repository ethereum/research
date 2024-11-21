from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like, Z, G
)
from utils import (
    log2, HALF, cp, reverse_bit_order,
    merkelize_top_dimension, get_challenges, rbo_index_to_original
)
from precomputes import folded_rbos, invx, invy
from fast_fft import fft
from merkle import merkelize, hash, get_branch, verify_branch

BASE_CASE_SIZE = 64
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND
NUM_CHALLENGES = 80

# This is the folding step in FRI, where you combine the evaluations at two
# sets of N coordinates each into evaluations at one set of N coordinates.
# We do three rounds of folding at a time, so each FRI step drops the degree
# by 8x
def fold(values, coeff, first_round):
    for i in range(FOLDS_PER_ROUND):
        full_len, half_len = values.shape[-1], values.shape[-1]//2
        left, right = values[::2], values[1::2]
        f0 = (left + right) * HALF
        if i == 0 and first_round:
            twiddle = (
                invy[full_len: full_len * 2]
                [folded_rbos[full_len:full_len*2:2]]
            )
        else:
            twiddle = (
                invx[full_len*2: full_len * 3]
                [folded_rbos[full_len:full_len*2:2]]
            )
        twiddle_box = zeros_like(left)
        twiddle_box[:] = twiddle.reshape((half_len,) + (1,) * (left.ndim-1))
        f1 = (left - right) * HALF * twiddle_box
        values = f0 + f1 * coeff
    return values

# This performs the same folding step as above, but at a pre-supplied list
# of positions. This is used in verification, where we repeat the same
# calculation as was done to make the proof, but only at a few randomly
# selected indices
def fold_with_positions(values, domain_size, positions, coeff, first_round):
    positions = positions[::2]
    for i in range(FOLDS_PER_ROUND):
        left, right = values[::2], values[1::2]
        f0 = (left + right) * HALF
        if i == 0 and first_round:
            unrbo_positions = rbo_index_to_original(domain_size, positions)
            twiddle = invy[domain_size + unrbo_positions]
        else:
            unrbo_positions = rbo_index_to_original(
                domain_size * 2,
                (positions << 1) >> i
            )
            twiddle = invx[domain_size * 2 + unrbo_positions]
        twiddle_box = zeros_like(left)
        twiddle_box[:] = twiddle.reshape((left.shape[0],) + (1,)*(left.ndim-1))
        f1 = (left - right) * HALF * twiddle_box
        values = f0 + f1 * coeff
        positions = positions[::2]
        domain_size //= 2
    return values

# Generate a FRI proof
def prove_low_degree(evaluations, extra_entropy=b''):
    assert evaluations.ndim == 1 and isinstance(evaluations, ExtendedM31)
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
        trees.append(merkelize_top_dimension(values.reshape(
            (len(values) // FOLD_SIZE_RATIO, FOLD_SIZE_RATIO)
            + values.shape[1:]
        )))
        roots.append(trees[-1][1])
        print('Root: 0x{}'.format(roots[-1].hex()))
        print("Descent round {}: {} values".format(i+1, len(values)))
        fold_factor = ExtendedM31(get_challenges(b''.join(roots), modulus, 4))
        print("Fold factor: {}".format(fold_factor))
        values = fold(values, fold_factor, i==0)
    entropy = extra_entropy + b''.join(roots) + values.tobytes()
    challenges = get_challenges(
        entropy, len(evaluations) >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    round_challenges = (
        challenges.reshape((1,)+challenges.shape)
        >> cp.arange(0, rounds * FOLDS_PER_ROUND, FOLDS_PER_ROUND)
        .reshape((rounds,) + (1,) * challenges.ndim)
    )

    branches = [
        [get_branch(tree, c) for c in r_challenges]
        for i, (r_challenges, tree) in enumerate(zip(round_challenges, trees))
    ]
    round_challenges_xfold = (
        round_challenges.reshape(round_challenges.shape + (1,)) * 8
        + cp.arange(FOLD_SIZE_RATIO).reshape(1, 1, FOLD_SIZE_RATIO)
    )

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

# Verify a FRI proof
def verify_low_degree(proof, extra_entropy=b''):
    roots = proof["roots"]
    branches = proof["branches"]
    leaf_values = proof["leaf_values"]
    final_values = proof["final_values"]
    len_evaluations = final_values.shape[0] << (FOLDS_PER_ROUND * len(roots))
    print("Verifying FRI proof")
    entropy = extra_entropy + b''.join(roots) + final_values.tobytes()
    challenges = get_challenges(
        entropy, len_evaluations >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    # Re-run the descent at the pseudorandomly-chosen set of points, and
    # verify consistency at each step
    for i in range(len(roots)):
        print("Descent round {}".format(i+1))
        fold_factor = ExtendedM31(
            get_challenges(b''.join(roots[:i+1]), modulus, 4)
        )
        print("Fold factor: {}".format(fold_factor))
        evaluation_size = len_evaluations >> (i * FOLDS_PER_ROUND)
        positions = (
            challenges.reshape((NUM_CHALLENGES, 1)) * FOLD_SIZE_RATIO
            + cp.arange(FOLD_SIZE_RATIO)
        ).reshape((NUM_CHALLENGES * FOLD_SIZE_RATIO))
        folded_values = fold_with_positions(
            leaf_values[i].reshape((-1,)),
            evaluation_size,
            positions,
            fold_factor,
            i==0
        )
        if i < len(roots) - 1:
            expected_values = (
                leaf_values[i+1][
                    cp.arange(NUM_CHALLENGES),
                    challenges % FOLD_SIZE_RATIO
                ]
            )
        else:
            expected_values = final_values[challenges]
        assert folded_values == expected_values
        # Also verify the Merkle branches
        for j, c in enumerate(cp.copy(challenges)):
            assert verify_branch(
                roots[i], c, leaf_values[i][j].tobytes(), branches[i][j]
            )
        challenges >>= FOLDS_PER_ROUND
    o = zeros_like(final_values)
    N = final_values.shape[0]
    o[rbo_index_to_original(N, cp.arange(N))] = final_values
    coeffs = fft(o, is_top_level=False)
    assert coeffs[N//2:] == 0
    return True
