EXPANSION_FACTOR = 8
NUM_CHALLENGES = 32
PACKING_FACTOR = 16

from binary_fields import BinaryFieldElement as B
from utils import log2
from optimized_utils import (
    pack16, extend, multilinear_poly_eval,
    bytestobits, bitstobytes, uint16s_to_bits, bits_to_uint16s,
    evaluation_tensor_product, bigbin_to_int, big_mul
)
from merkle import hash, merkelize, get_root, get_branch, verify_branch
import numpy as np

# Take a single bit from a packed element
def unpack_bit(item, bit):
    return (int(item) >> bit) & 1

def choose_row_length_and_count(log_evaluation_count):
    log_row_length = (log_evaluation_count + 2) // 2
    log_row_count = (log_evaluation_count - 1) // 2
    row_length = 1 << log_row_length
    row_count = 1 << log_row_count
    return log_row_length, log_row_count, row_length, row_count

# "Block-level encoding-based polynomial commitment scheme", section 3.11 of
# https://eprint.iacr.org/2023/1784.pdf
#
# An optimized implementation that tries to go as fast as you can in python,
# using numpy

def optimized_binius_proof(evaluations, evaluation_point):
    # Rearrange evaluations into a row_length * row_count grid
    assert len(evaluation_point) == log2(len(evaluations) * 8)
    log_row_length, log_row_count, row_length, row_count = \
        choose_row_length_and_count(log2(len(evaluations) * 8))
    # Directly treat the rows as a list of uint16's
    rows = (
        np.frombuffer(evaluations, dtype='<u2')
        .reshape((row_count, row_length // PACKING_FACTOR))
    )
    extended_rows = extend(rows, EXPANSION_FACTOR)
    extended_row_length = row_length * EXPANSION_FACTOR // PACKING_FACTOR

    # Compute t_prime, a linear combination of the rows
    # The linear combination is carefully chosen so that the evaluation of the
    # multilinear polynomial at the `evaluation_point` is itself either on
    # t_prime, or a linear combination of elements of t_prime
    row_combination = \
        evaluation_tensor_product(evaluation_point[log_row_length:])
    assert len(row_combination) == len(rows) == row_count
    t_prime = np.array([
        np.bitwise_xor.reduce([
            row_combination[i] for i in range(row_count) if
            int(rows[i][j//16]) & (1 << (j%16))
        ]) for j in range(row_length)
    ])

    # Pack columns into a Merkle tree, to commit to them
    columns = np.transpose(extended_rows)
    bytes_per_element = PACKING_FACTOR//8
    packed_columns = [col.tobytes('C') for col in columns]
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

def verify_optimized_binius_proof(proof):
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
        packed_column = col.tobytes('C')
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)

    # Use the same Reed-Solomon code that the prover used to extend the rows,
    # but to extend t_prime. We do this separately for each bit of t_prime
    t_prime_bit_length = 128
    t_prime_bits = uint16s_to_bits(t_prime)
    rows = bits_to_uint16s(np.transpose(t_prime_bits))
    extended_slices = extend(rows, EXPANSION_FACTOR)

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
            computed_tprime = np.bitwise_xor.reduce([
                row_combination[i] for i in range(row_count) if
                unpack_bit(column[i], b)
            ])
            expected_tprime = bits_to_uint16s(np.array([
                unpack_bit(x, b) for x in extended_slices[:, challenge]
            ], dtype=np.uint16))
            assert np.array_equal(expected_tprime, computed_tprime)
    print("T_prime matches Merkle branches")

    # Take the right linear combination of elements *within* t_prime to
    # extract the evaluation of the original multilinear polynomial at
    # the desired point
    col_combination = \
        evaluation_tensor_product(evaluation_point[:log_row_length])
    computed_eval = np.bitwise_xor.reduce(
        big_mul(t_prime, col_combination)
    )
    print(f"Testing evaluation: expected {value} computed {computed_eval}")
    assert np.array_equal(computed_eval, value)
    return True
