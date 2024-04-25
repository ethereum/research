EXPANSION_FACTOR = 8
NUM_CHALLENGES = 4
PACKING_FACTOR = 32

from binary_fields import BinaryFieldElement
from utils import get_class, eval_poly_at, mul_polys, compute_lagrange_poly, multilinear_poly_eval, extend, evaluation_tensor_product
from merkle import hash, merkelize, get_root, get_branch, verify_branch

# Suppose that we will be taking in `items` bits, and treating those bits as
# items // PACKING_FACTOR field elements. Now, imagine we Reed-Solomon extend
# those field elements. The index-coord bit of the extended value (ie. the
# index % PACKING_FACTOR bit of the index // PACKING_FACTOR field element) is
# a linear function of the input bits. Specifically, it must be a sum of a
# particular subset of the input bits.
#
# Which subset? The answer is independent of the input. We compute what that
# subset is, and return a vector that is `items` bits long, where the index-k
# bit represents whether or not bit k is part of that subset.

def generate_reed_solomon_matrix_row(items, coord):
    poly_width = items // PACKING_FACTOR
    evals = [
        eval_poly_at(compute_lagrange_poly(poly_width, BinaryFieldElement(i)), coord // PACKING_FACTOR)
        for i in range(poly_width)
    ]
    o = [None for _ in range(items)]
    for i in range(poly_width):
        value = evals[i]
        for bit in range(PACKING_FACTOR):
            o[i * PACKING_FACTOR + bit] = ((value * (1<<bit)).value >> (coord % PACKING_FACTOR)) & 1
    return o

# Generates the full matrix representing the operation of extending `items`
# bits to `items * expansion_factor` bits, by treating the input bits as
# packed field elements and Reed-Solomon extending them
def generate_reed_solomon_matrix(items, expansion_factor=2):
    return [
        generate_reed_solomon_matrix_row(items, coord)
        for coord in range(items, items * expansion_factor)
    ]

# Use the matrix row to actually compute a particular bit of the extension
def apply_reed_solomon_matrix_row(vals, row):
    return sum(
        [v for v,bit in zip(vals, row) if bit == 1],
        BinaryFieldElement(0)
    )

# Actually compute the extension
def extend_via_matrix(vals, matrix):
    return vals + [
        apply_reed_solomon_matrix_row(vals, row)
        for row in matrix
    ]

# Pack `PACKING_FACTOR` items into a field element
def pack(item):
    assert len(item) == PACKING_FACTOR
    o = BinaryFieldElement(0)
    for i, v in enumerate(item):
        o += v * (2**i)
    return o

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

    # Extend each row using a Reed-Solomon code
    # Difference with 3.7: here, we use a bitwise approach to extend the rows
    M = generate_reed_solomon_matrix(row_length, EXPANSION_FACTOR)
    extended_rows = [extend_via_matrix(row, M) for row in rows]
    extended_row_length = row_length * EXPANSION_FACTOR

    # Compute t_prime, a linear combination of the rows
    row_combination = evaluation_tensor_product(evaluation_point[L//2:])
    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum([rows[i][j] * row_combination[i] for i in range(row_count)], BinaryFieldElement(0))
        for j in range(row_length)
    ]

    # Pack columns into a Merkle tree, to commit to them
    columns = [
        [row[j:j+PACKING_FACTOR] for row in extended_rows]
        for j in range(0, extended_row_length, PACKING_FACTOR)
    ]
    bytes_per_element = PACKING_FACTOR//8
    packed_columns = [
        b''.join(pack(x).to_bytes(bytes_per_element, 'little') for x in col)
        for col in columns
    ]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)

    # Challenge in a few positions, to get branches
    packed_length = extended_row_length // PACKING_FACTOR
    challenges = [
        int.from_bytes(hash(root + bytes([i])), 'little') % packed_length
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
    L = len(evaluation_point)
    row_length = 1 << (L // 2)
    row_count = 1 << ((L + 1) // 2)
    extended_row_length = row_length * EXPANSION_FACTOR

    # Compute challenges. Should output the same as what prover computed
    packed_length = extended_row_length // PACKING_FACTOR
    challenges = [
        int.from_bytes(hash(root + bytes([i])), 'little') % packed_length
        for i in range(NUM_CHALLENGES)
    ]

    # Verify the correctness of the Merkle branches
    bytes_per_element = PACKING_FACTOR//8
    for challenge, branch, col in zip(challenges, branches, columns):
        packed_column = \
            b''.join(pack(x).to_bytes(bytes_per_element, 'little') for x in col)
        print(f"Verifying Merkle branch for column {challenge}")
        assert verify_branch(root, challenge, packed_column, branch)

    # Use the same Reed-Solomon code that the prover used to extend the rows,
    # but to extend t_prime
    M = generate_reed_solomon_matrix(row_length, EXPANSION_FACTOR)
    extended_t_prime = extend_via_matrix(t_prime, M)

    # Here, we take advantage of the linearity of the code. A linear combination
    # of the Reed-Solomon extension gives the same result as an extension of the
    # linear combination.
    row_combination = evaluation_tensor_product(proof["evaluation_point"][L // 2:])
    for column, challenge in zip(proof['columns'], challenges):
        for b in range(PACKING_FACTOR):
            computed_tprime = sum(
                [column[i][b] * row_combination[i] for i in range(row_count)],
                BinaryFieldElement(0)
            )
            expected_tprime = extended_t_prime[challenge * PACKING_FACTOR + b]
            assert expected_tprime == computed_tprime
    print("T_prime matches Merkle branches")

    # Take the right linear combination of elements *within* t_prime to
    # extract the evaluation of the original multilinear polynomial at
    # the desired point
    col_combination = evaluation_tensor_product(evaluation_point[:L // 2])
    computed_eval = sum(
        [t_prime[i] * col_combination[i] for i in range(row_length)],
        BinaryFieldElement(0)
    )
    print(f"Testing evaluation: expected {value} computed {computed_eval}")
    assert computed_eval == value
    return True
