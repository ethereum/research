from fft import (
    fft, inv_fft, get_initial_domain_of_size, log2,
    halve_domain, get_single_domain_value, halve_single_domain_value
)
from merkle import merkelize, hash, get_branch, verify_branch

BASE_CASE_SIZE = 128
FOLDS_PER_ROUND = 3
FOLD_SIZE_RATIO = 2**FOLDS_PER_ROUND
NUM_CHALLENGES = 80

def extend_trace(field, trace):
    small_domain = get_initial_domain_of_size(field, len(trace))
    coeffs = fft.fft(trace, small_domain)
    big_domain = get_initial_domain_of_size(field, len(trace)*2)
    return fft.inv_fft(trace, big_domain)

def line_function(P1, P2, domain):
    x1, y1 = P1
    x2, y2 = P2
    denominator = (x1 * y2 - x2 * y1)
    a = (y2 - y1) / denominator
    b = (x1 - x2) / denominator
    c = -(a * x1 + b * y1)
    return [a * x + b * y + c for x,y in domain]

def rbo(values):
    if len(values) == 1:
        return values
    L = rbo(values[::2])
    R = rbo(values[1::2])
    return L + R

def folded_reverse_bit_order(values):
    L = rbo(values[::2])
    R = rbo(values[1::2][::-1])
    o = [None for _ in values]
    o[::2] = L
    o[1::2] = R
    return o

undo_folded_reverse_bit_order = folded_reverse_bit_order

def rbo_index_to_original(length, index):
    sub = int(bin(length + index)[3:-1][::-1], 2)
    return sub * 2 if index % 2 == 0 else length - 1 - sub * 2

def fold(values, coeff, domain):
    for i in range(FOLDS_PER_ROUND):
        left, right = values[::2], values[1::2]
        f0 = [(L+R)/2 for L,R in zip(left, right)]
        if isinstance(domain[0], tuple):
            f1 = [(L-R)/(2*y) for L,R,(x,y) in zip(left, right, domain[::2])]
            # twiddle = [y.inv() for (x,y) in domain[::2]]
        else:
            f1 = [(L-R)/(2*x) for L,R,x in zip(left, right, domain[::2])]
            # twiddle = domain[::2]
        values = [f0val + coeff * f1val for f0val, f1val in zip(f0, f1)]
        domain = halve_domain(domain[::2], preserve_length=True)
    return values, domain


# Get the Merkle branch challenge indices from a root
def get_challenges(root, domain_size, num_challenges):
    challenge_data = b''.join(
        hash(root + bytes([i//256, i%256])) for i in range((num_challenges + 7) // 8)
    )
    return [
        int.from_bytes(challenge_data[i:i+4], 'little') % domain_size
        for i in range(0, num_challenges * 4, 4)
    ]

def is_rbo_low_degree(evaluations, domain):
    halflen = len(evaluations)//2
    return fft(
        undo_folded_reverse_bit_order(evaluations),
        undo_folded_reverse_bit_order(domain)
    )[halflen:] == [0] * halflen

def chunkify(values):
    return [
        b''.join(x.to_bytes() for x in values[i:i+FOLD_SIZE_RATIO])
        for i in range(0, len(values), FOLD_SIZE_RATIO)
    ]

def unchunkify(field, data):
    return [field.from_bytes(data[i:i+16]) for i in range(0, len(data), 16)]

def prove_low_degree(evaluations):
    # Input must already be in extension field
    assert hasattr(evaluations[0].__class__, 'subclass')
    E = evaluations[0].__class__
    M = E.subclass.modulus
    domain = folded_reverse_bit_order(
        get_initial_domain_of_size(E.subclass, len(evaluations))
    )
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
        fold_factor = E(get_challenges(b''.join(roots), M, 4))
        print("Fold factor: {}".format(fold_factor))
        values, domain = fold(values, fold_factor, domain)
    entropy = b''.join(roots + [x.to_bytes() for x in values])
    challenges = get_challenges(
        entropy, len(evaluations) >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    round_challenges = [
        [c >> (i * FOLDS_PER_ROUND) for c in challenges]
        for i in range(rounds)
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

def verify_low_degree(proof):
    roots = proof["roots"]
    branches = proof["branches"]
    leaf_values = proof["leaf_values"]
    final_values = proof["final_values"]
    assert hasattr(final_values[0].__class__, 'subclass')
    E = final_values[0].__class__
    M = E.subclass.modulus
    len_evaluations = len(final_values) << (FOLDS_PER_ROUND * len(roots))
    print("Verifying proof")
    # Prove descent
    entropy = b''.join(roots + [x.to_bytes() for x in final_values])
    challenges = get_challenges(
        entropy, len_evaluations >> FOLDS_PER_ROUND, NUM_CHALLENGES
    )
    for i in range(len(roots)):
        print("Descent round {}".format(i+1))
        fold_factor = E(get_challenges(b''.join(roots[:i+1]), M, 4))
        print("Fold factor: {}".format(fold_factor))
        evaluation_size = len_evaluations >> (i * FOLDS_PER_ROUND)
        positions = sum([
            list(range(c*FOLD_SIZE_RATIO, (c+1)*FOLD_SIZE_RATIO))
            for c in challenges
        ], [])
        if i == 0:
            domain = [
                get_single_domain_value(
                    E.subclass,
                    evaluation_size,
                    rbo_index_to_original(evaluation_size, pos)
                ) for pos in positions
            ]
        else:
            domain = [
                halve_single_domain_value(get_single_domain_value(
                    E.subclass,
                    evaluation_size * 2,
                    rbo_index_to_original(evaluation_size * 2, pos * 2)
                )) for pos in positions
            ]
        folded_values, _ = fold(sum(leaf_values[i], []), fold_factor, domain)
        if i < len(roots) - 1:
            expected_values = [
                leaf_values[i+1][j][c % FOLD_SIZE_RATIO]
                for j,c in enumerate(challenges)
            ]
        else:
            expected_values = [final_values[c] for c in challenges]
        assert folded_values == expected_values
        for j, c in enumerate(challenges):
            assert verify_branch(
                roots[i], c, chunkify(leaf_values[i][j])[0], branches[i][j]
            )
        challenges = [c >> FOLDS_PER_ROUND for c in challenges]
    final_domain = folded_reverse_bit_order(
        halve_domain(
            get_initial_domain_of_size(E.subclass, len(final_values)*2)
        )
    )
    assert is_rbo_low_degree(final_values, final_domain)
    return True
