# Evaluates a polynomial, expressed as an array where element i is the
# ith degree term, at coordinate x, in the prime field with the given
# modulus
def eval_poly_at(poly, x, modulus):
    o = 0
    for p, v in enumerate(poly):
        o += v * pow(x, p, modulus) 
    return o % modulus

# Evaluates a polynomial for every coordinate in the given prime field
def eval_across_field(poly, modulus):
    return [eval_poly_at(poly, i, modulus) for i in range(modulus)]

# Interprets the given polynomial as a 2D polynomial (poly mod x^subdeg - y)
# and evaluates it at the given (x, y) coordinate
# eg. poly = [1, 2, 3, 4], subdeg = 2, the 2D polynomial becomes
# 4xy + 3y + 2x + 1
def eval_2d_poly_at(poly, x, y, subdeg, modulus):
    o = 0
    for p, v in enumerate(poly):
        o += v * pow(x, p % subdeg, modulus) * pow(y, p // subdeg, modulus)
    return o % modulus

# Interprets the given polynomial as a 2D polynomial, and evaluates it
# across the entire field x field square
def eval_across_square(poly, max_x, max_y, subdeg, modulus):
    o = []
    for y in range(max_y):
        p = []
        for x in range(max_x):
            p.append(eval_2d_poly_at(poly, x, y, subdeg, modulus))
        o.append(p)
    return o

# Recovers the polynomial that has the given y coordinates at the given
# x coordinates
def lagrange_interp(xs, ys, modulus):
    # Generate master numerator polynomial
    root = [1]
    for i in range(len(xs)):
        root.insert(0, 0)
        for j in range(len(root)-1):
            root[j] = (root[j] - root[j+1] * xs[i]) % modulus
    # Generate per-value numerator polynomials by dividing the master
    # polynomial back by each x coordinate
    nums = []
    for i in range(len(xs)):
        output = []
        last = 1
        for j in range(2,len(root)+1):
            output.insert(0, last)
            if j != len(root):
                last = root[-j] + last * xs[i]
            last = last % modulus
        nums.append(output)
    # Generate denominators by evaluating numerator polys at their x
    denoms = []
    for i in range(len(xs)):
        denom = 0
        xcpower = 1
        for j in range(len(nums[i])):
            denom += xcpower * nums[i][j]
            xcpower *= xs[i]
        denoms.append(denom % modulus)
    # Derive output
    b = [0] * len(xs)
    for i in range(len(xs)):
        yslice = ys[i] * pow(denoms[i], modulus - 2, modulus)
        for j in range(len(b)):
            b[j] += nums[i][j] * yslice
    return [x % modulus for x in b]

# Is the number a perfect square?
def is_perf_square(n): return n ** 0.5 == int(n** 0.5)

# Makes a low-degree proximity proof for the given polynomial
# by simply converting it into a 2D polynomial, with degree
# equal to the sqrt of the original degree - 1 and evaluating
# it at all (x, y) coordinates. The "diagonal"
# Q[i][i ** int(deg_lt ** 0.5)] for all i in the field is
# equivalent to the original data.
def mk_quadratic_proof(data, deg_lt, modulus):
    # Derive the polynomial from the data
    poly = lagrange_interp(range(len(data)), data, modulus)
    # Check that the polynomial actually is low-degree
    for i in range(deg_lt, len(poly)):
        assert poly[i] == 0
    # Max degree must be a perfect square
    assert is_perf_square(deg_lt)
    # Evaluate it across the entire square
    sq = eval_across_square(poly, modulus, modulus, int(deg_lt ** 0.5), modulus)
    return sq

# Checks the correctness of the above proof
def check_quadratic_proof(data, sq, deg_lt, checks, modulus):
    import random
    subdeg = int(deg_lt ** 0.5)
    for _ in range(checks):
        # Select a row and the corresponding column (column = row ** subdeg % modulus)
        check_col = random.randrange(len(sq))
        check_row = pow(check_col, subdeg, modulus)
        # Pick `subdeg` random cells in the same row
        row_cells = [(col, sq[check_row][col]) for col in sorted(range(modulus), key=lambda x: random.random())[:subdeg]]
        # Derive the polynomial (should be degree subdeg - 1)
        row_poly = lagrange_interp([x[0] for x in row_cells], [x[1] for x in row_cells], modulus)
        # Pick `subdeg` random cells in the same column
        col_cells = [(row, sq[row][check_col]) for row in sorted(range(modulus), key=lambda x: random.random())[:subdeg]]
        # Derive the polynomial (should be degree subdeg - 1)
        col_poly = lagrange_interp([x[0] for x in col_cells], [x[1] for x in col_cells], modulus)
        print('row %d eval' % check_row, eval_poly_at(row_poly, check_col, modulus))
        print('col %d eval' % check_col, eval_poly_at(col_poly, check_row, modulus))
        print('diag_in_sq', sq[check_row][check_col])
        print('in data', data[check_col])
        # Evaluate the polynomials along the "diagonal" (x, x ** subdeg), and check that the evaluated
        # polynomials, the values in the square along the diagonal, and the original data all match
        assert eval_poly_at(row_poly, check_col, modulus) == eval_poly_at(col_poly, check_row, modulus) == \
            sq[check_row][check_col] == data[check_col]
    return True

# Make a single-column low-degree proof
def mk_column_proof(data, deg_lt, modulus):
    import random
    # Derive the polynomial from the data
    poly = lagrange_interp(range(len(data)), data, modulus)
    # Check that the polynomial actually is low-degree
    for i in range(deg_lt, len(poly)):
        assert poly[i] == 0
    # Max degree must be a perfect square
    assert is_perf_square(deg_lt)
    # Max degree of the 2D polynomial (as in, the actual degree must be *less* than this)
    subdeg = int(deg_lt ** 0.5)
    # The order of the multiplicative group in the field (ie. p-1) must be a multiple
    # of the subdeg, so that there are only (modulus - 1) // subdeg values of x ** subdeg
    assert (modulus - 1) % subdeg == 0
    # All possible values of x ** subdeg
    admissible_rows = [x for x in range(modulus) if pow(x, (modulus - 1) // subdeg, modulus) == 1]
    assert len(admissible_rows) == modulus // subdeg
    # Select a random x coordinate
    xcor = random.randrange(modulus)
    # Return a column consisting of the evaluations of the 2D polynomial at all admissible y
    # coordinates
    return (xcor, [eval_2d_poly_at(poly, xcor, row, subdeg, modulus) for row in admissible_rows])

# Check the above generated proof
def check_column_proof(data, proof, deg_lt, checks, modulus):
    import random
    subdeg = int(deg_lt ** 0.5)
    check_col, column = proof
    # All possible values of x ** subdeg
    admissible_rows = [x for x in range(modulus) if pow(x, (modulus - 1) // subdeg, modulus) == 1]
    for i in range(checks):
        # Choose a random row to check
        check_row = random.choice(admissible_rows)
        print('Checking row %d' % check_row)
        # Get the x coordinates that satisfy x ** subdeg % modulus == check_row.
        # There are `subdeg` of these.
        xs = [x for x in range(modulus) if pow(x, subdeg, modulus) == check_row]
        print('Taking columns from data:', [(x, data[x]) for x in xs])
        assert len(xs) == subdeg
        # Interpolate a degree subdeg-1 polynomial from the above
        row_poly = lagrange_interp(xs, [data[x] for x in xs], modulus)
        print('Eval', eval_poly_at(row_poly, check_col, modulus))
        print('Actual', column[admissible_rows.index(check_row)])
        # Evaluate the polynomial at the x coordinate of the column. Check that
        # the value is the same as the value provided
        assert eval_poly_at(row_poly, check_col, modulus) == column[admissible_rows.index(check_row)]
    # Check that the column itself is low degree
    column_poly = lagrange_interp(admissible_rows, column, modulus)
    for i in range(subdeg+1, len(column_poly)):
        assert column_poly[i] == 0
    return True
