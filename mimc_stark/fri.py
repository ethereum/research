from permuted_tree import merkelize, mk_branch, verify_branch, mk_multi_branch, verify_multi_branch
from utils import get_power_cycle, get_pseudorandom_indices
from poly_utils import PrimeField

# Generate an FRI proof that the polynomial that has the specified
# values at successive powers of the specified root of unity has a
# degree lower than maxdeg_plus_1
#
# We use maxdeg+1 instead of maxdeg because it's more mathematically
# convenient in this case.

def prove_low_degree(values, root_of_unity, maxdeg_plus_1, modulus, exclude_multiples_of=0):
    f = PrimeField(modulus)
    print('Proving %d values are degree <= %d' % (len(values), maxdeg_plus_1))

    # If the degree we are checking for is less than or equal to 32,
    # use the polynomial directly as a proof
    if maxdeg_plus_1 <= 16:
        print('Produced FRI proof')
        return [[x.to_bytes(32, 'big') for x in values]]

    # Calculate the set of x coordinates
    xs = get_power_cycle(root_of_unity, modulus)
    assert len(values) == len(xs)

    # Put the values into a Merkle tree. This is the root that the
    # proof will be checked against
    m = merkelize(values)

    # Select a pseudo-random x coordinate
    special_x = int.from_bytes(m[1], 'big') % modulus

    # Calculate the "column" at that x coordinate
    # (see https://vitalik.ca/general/2017/11/22/starks_part_2.html)
    # We calculate the column by Lagrange-interpolating each row, and not
    # directly from the polynomial, as this is more efficient
    quarter_len = len(xs)//4
    x_polys = f.multi_interp_4(
        [[xs[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)],
        [[values[i+quarter_len*j] for j in range(4)] for i in range(quarter_len)]
    )
    column = [f.eval_quartic(p, special_x) for p in x_polys]
    m2 = merkelize(column)

    # Pseudo-randomly select y indices to sample
    ys = get_pseudorandom_indices(m2[1], len(column), 40, exclude_multiples_of=exclude_multiples_of)

    # Compute the positions for the values in the polynomial
    poly_positions = sum([[y + (len(xs) // 4) * j for j in range(4)] for y in ys], [])

    # This component of the proof, including Merkle branches
    o = [m2[1], mk_multi_branch(m2, ys), mk_multi_branch(m, poly_positions)]

    # Recurse...
    return [o] + prove_low_degree(column, f.exp(root_of_unity, 4),
                                  maxdeg_plus_1 // 4, modulus, exclude_multiples_of=exclude_multiples_of)

# Verify an FRI proof
def verify_low_degree_proof(merkle_root, root_of_unity, proof, maxdeg_plus_1, modulus, exclude_multiples_of=0):
    f = PrimeField(modulus)

    # Calculate which root of unity we're working with
    testval = root_of_unity
    roudeg = 1
    while testval != 1:
        roudeg *= 2
        testval = (testval * testval) % modulus

    # Powers of the given root of unity 1, p, p**2, p**3 such that p**4 = 1
    quartic_roots_of_unity = [1,
                              f.exp(root_of_unity, roudeg // 4),
                              f.exp(root_of_unity, roudeg // 2),
                              f.exp(root_of_unity, roudeg * 3 // 4)]

    # Verify the recursive components of the proof
    for prf in proof[:-1]:
        root2, column_branches, poly_branches = prf
        print('Verifying degree <= %d' % maxdeg_plus_1)

        # Calculate the pseudo-random x coordinate
        special_x = int.from_bytes(merkle_root, 'big') % modulus

        # Calculate the pseudo-randomly sampled y indices
        ys = get_pseudorandom_indices(root2, roudeg // 4, 40,
                                      exclude_multiples_of=exclude_multiples_of)

        # Compute the positions for the values in the polynomial
        poly_positions = sum([[y + (roudeg // 4) * j for j in range(4)] for y in ys], [])

        # Verify Merkle branches
        column_values = verify_multi_branch(root2, ys, column_branches)
        poly_values = verify_multi_branch(merkle_root, poly_positions, poly_branches)

        # For each y coordinate, get the x coordinates on the row, the values on
        # the row, and the value at that y from the column
        xcoords = []
        rows = []
        columnvals = []
        for i, y in enumerate(ys):
            # The x coordinates from the polynomial
            x1 = f.exp(root_of_unity, y)
            xcoords.append([(quartic_roots_of_unity[j] * x1) % modulus for j in range(4)])

            # The values from the original polynomial
            row = [int.from_bytes(x, 'big') for x in poly_values[i*4: i*4+4]]
            rows.append(row)

            columnvals.append(int.from_bytes(column_values[i], 'big'))

        # Verify for each selected y coordinate that the four points from the
        # polynomial and the one point from the column that are on that y 
        # coordinate are on the same deg < 4 polynomial
        polys = f.multi_interp_4(xcoords, rows)

        for p, c in zip(polys, columnvals):
            assert f.eval_quartic(p, special_x) == c

        # Update constants to check the next proof
        merkle_root = root2
        root_of_unity = f.exp(root_of_unity, 4)
        maxdeg_plus_1 //= 4
        roudeg //= 4

    # Verify the direct components of the proof
    data = [int.from_bytes(x, 'big') for x in proof[-1]]
    print('Verifying degree <= %d' % maxdeg_plus_1)
    assert maxdeg_plus_1 <= 16

    # Check the Merkle root matches up
    mtree = merkelize(data)
    assert mtree[1] == merkle_root

    # Check the degree of the data
    powers = get_power_cycle(root_of_unity, modulus)
    if exclude_multiples_of:
        pts = [x for x in range(len(data)) if x % exclude_multiples_of]
    else:
        pts = range(len(data))

    poly = f.lagrange_interp([powers[x] for x in pts[:maxdeg_plus_1]],
                             [data[x] for x in pts[:maxdeg_plus_1]])
    for x in pts[maxdeg_plus_1:]:
        assert f.eval_poly_at(poly, powers[x]) == data[x]   

    print('FRI proof verified')
    return True
