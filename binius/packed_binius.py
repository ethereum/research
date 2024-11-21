EXPANSION_FACTOR = 8
NUM_CHALLENGES = 32
PACKING_FACTOR = 16

from binary_fields import BinaryFieldElement
from utils import (
    get_class, pack_vector,
    evaluation_tensor_product, log2, multilinear_poly_eval
)
from merkle import hash, merkelize, get_root, get_branch, verify_branch
from binary_ntt import extend

# Take a single bit from a packed element
def unpack_bit(item, bit):
    return (item.value >> bit) & 1

def extend_row_of_bits(row):
    return extend(pack_vector(row, PACKING_FACTOR), EXPANSION_FACTOR)

def choose_row_length_and_count(log_evaluation_count):
    log_row_length = (log_evaluation_count + 2) // 2
    log_row_count = (log_evaluation_count - 1) // 2
    row_length = 1 << log_row_length
    row_count = 1 << log_row_count
    return log_row_length, log_row_count, row_length, row_count

# "Block-level encoding-based polynomial commitment scheme", section 3.11 of
# https://eprint.iacr.org/2023/1784.pdf
#
# This algorithm is more complex, but it closely follows a similar pattern to
# the simple scheme. Note that here, we have to be more opinionated with fields:
# evaluations within the hypercube MUST be 0 or 1, and for the Reed-Solomon code
# and the evaluation point we use binary fields

def packed_binius_proof(evaluations, evaluation_point):
    # Rearrange evaluations into a row_length * row_count grid
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(log2(len(evaluations)))
    rows = [
        evaluations[i:i+row_length]
        for i in range(0, len(evaluations), row_length)
    ]

    # Extend each row using a Reed-Solomon code
    # Difference with 3.7: here, we take the input as bits, and treat each
    # slice of `PACKING_FACTOR` bits as a single field element
    extended_rows = [extend_row_of_bits(row) for row in rows]
    extended_row_length = row_length * EXPANSION_FACTOR // PACKING_FACTOR

    # Compute t_prime, a linear combination of the rows
    # The linear combination is carefully chosen so that the evaluation of the
    # multilinear polynomial at the `evaluation_point` is itself either on
    # t_prime, or a linear combination of elements of t_prime
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum(
            [row_combination[i] for i in range(row_count) if rows[i][j] == 1],
            BinaryFieldElement(0)
        ) for j in range(row_length)
    ]

    # Pack columns into a Merkle tree, to commit to them
    columns = [
        [row[j] for row in extended_rows]
        for j in range(extended_row_length)
    ]
    bytes_per_element = PACKING_FACTOR//8
    packed_columns = [
        b''.join(x.to_bytes(bytes_per_element, 'little') for x in col)
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

def verify_packed_binius_proof(proof):
    columns, evaluation_point, value, t_prime, root, branches = (
        proof['columns'],
        proof['evaluation_point'],
        proof['eval'],
        proof['t_prime'],
        proof['root'],
        proof['branches'],
    )

    # Compute the row length and row count of the grid. Should output same
    # numbers as what prover gave
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(len(evaluation_point))
    extended_row_length = row_length * EXPANSION_FACTOR // PACKING_FACTOR

    # Compute challenges. Should output the same as what prover computed
    challenges = [
        int.from_bytes(hash(root + bytes([i])), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]

    # Verify the correctness of the Merkle branches
    bytes_per_element = PACKING_FACTOR//8
    for challenge, branch, col in zip(challenges, branches, columns):
        packed_column = \
            b''.join(x.to_bytes(bytes_per_element, 'little') for x in col)
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)

    # Use the same Reed-Solomon code that the prover used to extend the rows,
    # but to extend t_prime. We do this separately for each bit of t_prime
    t_prime_bit_length = max(x.bit_length() for x in evaluation_point)
    extended_slices = [
        extend_row_of_bits([unpack_bit(v, i) for v in t_prime])
        for i in range(t_prime_bit_length)
    ]

    # Here, we take advantage of the linearity of the code. A linear combination
    # of the Reed-Solomon extension gives the same result as an extension of the
    # linear combination.
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    for column, challenge in zip(columns, challenges):
        # Each extended row, at the column we're querying, contains a field
        # element `PACKING_FACTOR` bits wide. We treat each bit separately
        for b in range(PACKING_FACTOR):
            # We apply the same linear combination to that sub-column of bits
            # that the prover applied to generate t_prime
            computed_tprime = sum(
                [
                    row_combination[i] * unpack_bit(column[i], b)
                    for i in range(row_count)
                ],
                BinaryFieldElement(0)
            )
            expected_tprime = pack_vector([
                unpack_bit(_slice[challenge], b)
                for _slice in extended_slices
            ], 128)[0]
            assert expected_tprime == computed_tprime
    print("T_prime matches Merkle branches")

    # Take the right linear combination of elements *within* t_prime to
    # extract the evaluation of the original multilinear polynomial at
    # the desired point
    col_combination = \
        evaluation_tensor_product(evaluation_point[:log_row_length])
    computed_eval = sum(
        [t_prime[i] * col_combination[i] for i in range(row_length)],
        BinaryFieldElement(0)
    )
    print(f"Testing evaluation: expected {value} computed {computed_eval}")
    assert computed_eval == value
    return True
