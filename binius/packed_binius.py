EXPANSION_FACTOR = 8
NUM_CHALLENGES = 4
PACKING_FACTOR = 32

from binary_fields import BinaryFieldElement
from utils import get_class, eval_poly_at, mul_polys, compute_lagrange_poly, multilinear_poly_eval, extend, evaluation_tensor_product, Vector, zero_of_same_type
from merkle import hash, merkelize, get_root, get_branch, verify_branch

# "Block-level encoding-based polynomial commitment scheme", section 3.11 of
# https://eprint.iacr.org/2023/1784.pdf
#
# This algorithm is more complex, but it closely follows a similar pattern to
# the simple scheme. Note that here, we have to be more opinionated with fields:
# evaluations within the hypercube MUST be 0 or 1, and for the Reed-Solomon code
# and the evaluation point we use binary fields

def packed_binius_proof(evaluations, evaluation_point):

    # Rearrange evaluations into a row_length * row_count grid
    L = len(evaluations).bit_length() - 1
    row_length = 1 << (L // 2)
    row_count = 1 << ((L + 1) // 2)
    rows = [evaluations[i:i+row_length] for i in range(0, len(evaluations), row_length)]

    # Difference with 3.7: here, we group the bits within a row into blocks of size
    # PACKING_FACTOR each
    packed_rows = [
        [Vector(row[i:i+PACKING_FACTOR]) for i in range(0, len(row), PACKING_FACTOR)]
        for row in rows
    ]
    packed_row_length = row_length // PACKING_FACTOR

    # We extend the packed groups of bits
    extended_rows = [extend(row, expansion_factor=EXPANSION_FACTOR) for row in packed_rows]
    extended_row_length = packed_row_length * EXPANSION_FACTOR
    row_combination = evaluation_tensor_product(evaluation_point[L//2:])

    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum([rows[i][j] * row_combination[i] for i in range(row_count)], BinaryFieldElement(0))
        for j in range(row_length)
    ]
    columns = [[row[j] for row in extended_rows] for j in range(extended_row_length)]
    bytes_per_element = 1
    while 256**bytes_per_element < row_length:
        bytes_per_element *= 2
    packed_columns = [b''.join(x.to_bytes(bytes_per_element, 'little') for x in col) for col in columns]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)
    challenges = [
        int.from_bytes(hash(root + i.to_bytes(4, 'little')), 'little') % extended_row_length
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
    # Check Merkle branches
    root = proof["root"]
    extended_row_length = 2**len(proof["branches"][0])
    challenges = [
        int.from_bytes(hash(root + i.to_bytes(4, 'little')), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]
    packed_row_length = extended_row_length // EXPANSION_FACTOR
    row_length = packed_row_length * PACKING_FACTOR
    row_count = len(proof['columns'][0])
    bytes_per_element = 1
    while 256**bytes_per_element < packed_row_length:
        bytes_per_element *= 2
    for challenge, branch, column in zip(challenges, proof['branches'], proof['columns']):
        packed_column = b''.join(x.to_bytes(bytes_per_element, 'little') for x in column)
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)
    t_prime = proof["t_prime"]
    packed_t_prime = [Vector(t_prime[i:i+PACKING_FACTOR]) for i in range(0, len(t_prime), PACKING_FACTOR)]
    extended_t_prime = extend(packed_t_prime, expansion_factor=EXPANSION_FACTOR)
    log_col_count = row_length.bit_length() - 1
    print('log col count', log_col_count)
    row_combination = evaluation_tensor_product(proof["evaluation_point"][log_col_count:])
    for column, challenge in zip(proof['columns'], challenges):
        print(len(column), len(row_combination), row_count)
        expected_tprime = sum([column[i] * row_combination[i] for i in range(row_count)], zero_of_same_type(column[0]))
        print(f"Testing challenge on column {challenge}: expected {expected_tprime} computed {extended_t_prime[challenge]}")
        assert expected_tprime == extended_t_prime[challenge]
    col_combination = evaluation_tensor_product(proof["evaluation_point"][:log_col_count])
    print('col comb', col_combination)
    computed_eval = sum([proof["t_prime"][i] * col_combination[i] for i in range(row_length)], BinaryFieldElement(0))
    print(f"Testing evaluation: expected {proof['eval']} computed {computed_eval}")
    assert computed_eval == proof["eval"]
    return True
