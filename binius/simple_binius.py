EXPANSION_FACTOR = 8
NUM_CHALLENGES = 32
BYTES_PER_ELEMENT = 2

from utils import (
    get_class, enforce_type_compatibility,
    evaluation_tensor_product, log2, multilinear_poly_eval
)
from binary_ntt import extend
from merkle import hash, merkelize, get_root, get_branch, verify_branch

def choose_row_length_and_count(log_evaluation_count):
    log_row_length = log_evaluation_count // 2
    log_row_count = (log_evaluation_count + 1) // 2
    row_length = 1 << log_row_length
    row_count = 1 << log_row_count
    return log_row_length, log_row_count, row_length, row_count

# An implementation of the "Basic small-field construction", construction 3.7 of
# https://eprint.iacr.org/2023/1784.pdf

def simple_binius_proof(evaluations, evaluation_point):
    cls, evaluations, evaluation_point = \
        enforce_type_compatibility(evaluations, evaluation_point)

    # Rearrange evaluations into a row_length * row_count grid
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(log2(len(evaluations)))
    rows = [
        evaluations[i:i+row_length]
        for i in range(0, len(evaluations), row_length)
    ]

    # Extend each row using a Reed-Solomon code
    extended_rows = [extend(row, EXPANSION_FACTOR) for row in rows]
    extended_row_length = row_length * EXPANSION_FACTOR

    # Compute t_prime, a linear combination of the rows
    # The linear combination is carefully chosen so that the evaluation of the
    # multilinear polynomial at the `evaluation_point` is itself either on
    # t_prime, or a linear combination of elements of t_prime
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum([rows[i][j] * row_combination[i] for i in range(row_count)], cls(0))
        for j in range(row_length)
    ]

    # Pack columns into a Merkle tree, to commit to them
    columns = [
        [row[j] for row in extended_rows]
        for j in range(extended_row_length)
    ]
    packed_columns = [
        b''.join(x.to_bytes(BYTES_PER_ELEMENT, 'little') for x in col)
        for col in columns
    ]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)

    # Challenge in a few positions, to get branches
    challenges = [
        int.from_bytes(hash(root + bytes([i])), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]
    return {
        'root': root,
        'evaluation_point': evaluation_point,
        'eval': multilinear_poly_eval(evaluations, evaluation_point),
        't_prime': t_prime,
        'columns': [columns[c] for c in challenges],
        'branches': [get_branch(merkle_tree, c) for c in challenges],
    }

def verify_simple_binius_proof(proof):
    cls, columns, evaluation_point, value, t_prime = enforce_type_compatibility(
        proof['columns'],
        proof['evaluation_point'],
        proof['eval'],
        proof['t_prime']
    )
    root, branches = proof["root"], proof["branches"]

    # Compute the row length and row count of the grid. Should output same
    # numbers as what prover gave
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(len(evaluation_point))
    extended_row_length = row_length * EXPANSION_FACTOR

    # Compute challenges. Should output the same as what prover computed
    challenges = [
        int.from_bytes(hash(root + bytes([i])), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]

    # Verify the correctness of the Merkle branches
    for challenge, branch, column in zip(challenges, branches, columns):
        packed_column = \
            b''.join(x.to_bytes(BYTES_PER_ELEMENT, 'little') for x in column)
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)

    # Use the same Reed-Solomon code that the prover used to extend the rows,
    # but to extend t_prime
    extended_t_prime = extend(proof["t_prime"], EXPANSION_FACTOR)

    # Here, we take advantage of the linearity of the code. A linear combination
    # of the Reed-Solomon extension gives the same result as an extension of the
    # linear combination.
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    for column, challenge in zip(proof['columns'], challenges):
        expected_tprime = sum(
            [column[i] * row_combination[i] for i in range(row_count)],
            cls(0)
        )
        print(
            f"Testing challenge on column {challenge}: expected "
            f"{expected_tprime} computed {extended_t_prime[challenge]}"
        )
        assert expected_tprime == extended_t_prime[challenge]

    # Take the right linear combination of elements *within* t_prime to
    # extract the evaluation of the original multilinear polynomial at
    # the desired point
    col_combination = \
        evaluation_tensor_product(evaluation_point[:log_row_length])
    computed_eval = sum(
        [t_prime[i] * col_combination[i] for i in range(row_length)],
        cls(0)
    )
    print(f"Testing evaluation: expected {value} computed {computed_eval}")
    assert computed_eval == value
    return True
