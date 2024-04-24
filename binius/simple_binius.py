EXPANSION_FACTOR = 8
NUM_CHALLENGES = 4

from utils import get_class, eval_poly_at, mul_polys, compute_lagrange_poly, multilinear_poly_eval, extend, evaluation_tensor_product
from merkle import hash, merkelize, get_root, get_branch, verify_branch

def simple_binius_proof(evals, evaluation_point):
    cls = get_class([evals, evaluation_point])
    L = len(evals).bit_length() - 1
    row_length = 1 << (L // 2)
    row_count = 1 << ((L + 1) // 2)
    rows = [evals[i:i+row_length] for i in range(0, len(evals), row_length)]
    extended_rows = [extend(row, expansion_factor=EXPANSION_FACTOR) for row in rows]
    extended_row_length = row_length * EXPANSION_FACTOR
    row_combination = evaluation_tensor_product(evaluation_point[L//2:])

    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum([rows[i][j] * row_combination[i] for i in range(row_count)], cls(0))
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
        'eval': multilinear_poly_eval(evals, evaluation_point),
        't_prime': t_prime,
        'columns': [columns[c] for c in challenges],
        'branches': [get_branch(merkle_tree, c) for c in challenges],
    }

def verify_simple_binius_proof(proof):
    cls = get_class([proof['columns'], proof['evaluation_point'], proof['eval']])
    # Check Merkle branches
    root = proof["root"]
    extended_row_length = 2**len(proof["branches"][0])
    challenges = [
        int.from_bytes(hash(root + i.to_bytes(4, 'little')), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]
    row_length = extended_row_length // EXPANSION_FACTOR
    row_count = len(proof['columns'][0])
    bytes_per_element = 1
    while 256**bytes_per_element < row_length:
        bytes_per_element *= 2
    for challenge, branch, column in zip(challenges, proof['branches'], proof['columns']):
        packed_column = b''.join(x.to_bytes(bytes_per_element, 'little') for x in column)
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)
    extended_t_prime = extend(proof["t_prime"], expansion_factor=EXPANSION_FACTOR)
    log_col_count = row_length.bit_length() - 1
    print('log col count', log_col_count)
    row_combination = evaluation_tensor_product(proof["evaluation_point"][log_col_count:])
    for column, challenge in zip(proof['columns'], challenges):
        expected_tprime = sum([column[i] * row_combination[i] for i in range(row_count)], cls(0))
        print(f"Testing challenge on column {challenge}: expected {expected_tprime} computed {extended_t_prime[challenge]}")
        assert expected_tprime == extended_t_prime[challenge]
    col_combination = evaluation_tensor_product(proof["evaluation_point"][:log_col_count])
    print('col comb', col_combination)
    computed_eval = sum([proof["t_prime"][i] * col_combination[i] for i in range(row_length)], cls(0))
    print(f"Testing evaluation: expected {proof['eval']} computed {computed_eval}")
    assert computed_eval == proof["eval"]
    return True
