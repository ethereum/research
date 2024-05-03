EXPANSION_FACTOR = 8
NUM_CHALLENGES = 32
PACKING_FACTOR = 16

from binary_fields import BinaryFieldElement as B
from utils import log2
from optimized_utils import (
    extend, multilinear_poly_eval,
    bytestobits, bitstobytes, uint16s_to_bits, bits_to_uint16s,
    evaluation_tensor_product, bigbin_to_int, big_mul, multisubset
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
    rows_as_bits = uint16s_to_bits(rows)
    t_prime = multisubset(row_combination, rows_as_bits.transpose())

    # Pack columns into a Merkle tree, to commit to them
    columns = np.transpose(extended_rows)
    bytes_per_element = PACKING_FACTOR//8
    packed_columns = [col.tobytes('C') for col in columns]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)

    # Challenge in a few positions, to get branches
    challenges = np.array([
        int.from_bytes(hash(root + bytes([i])), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ], dtype=np.uint16)
    return {
        'root': root,
        'evaluation_point': evaluation_point,
        'eval': multilinear_poly_eval(evaluations, evaluation_point),
        't_prime': t_prime,
        'columns': columns[challenges],
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
    challenges = np.array([
        int.from_bytes(hash(root + bytes([i])), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ], dtype=np.uint16)

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
    column_bits = uint16s_to_bits(columns[..., np.newaxis])
    computed_tprimes = multisubset(
        row_combination,
        np.transpose(column_bits, (0,2,1))
    )
    extended_slices_bits = uint16s_to_bits(
        extended_slices[:, challenges, np.newaxis]
    )
    assert np.array_equal(
        computed_tprimes,
        bits_to_uint16s(np.transpose(extended_slices_bits, (1, 2, 0)))
    )
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
