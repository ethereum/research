modulus_poly = [1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 1, 0, 0, 1,
                1]
modulus_poly_as_int = sum([(v << i) for i, v in enumerate(modulus_poly)])
degree = len(modulus_poly) - 1

two_to_the_degree = 2**degree
two_to_the_degree_m1 = 2**degree - 1

def galoistpl(a):
    # 2 is not a primitive root, so we have to use 3 as our logarithm base
    if a * 2 < two_to_the_degree:
        return (a * 2) ^ a
    else:
        return (a * 2) ^ a ^ modulus_poly_as_int

# Precomputing a log table for increased speed of addition and multiplication
glogtable = [0] * (two_to_the_degree)
gexptable = []
v = 1
for i in range(two_to_the_degree_m1):
    glogtable[v] = i
    gexptable.append(v)
    v = galoistpl(v)

gexptable += gexptable + gexptable

# Add two values in the Galois field
def galois_add(x, y):
    return x ^ y

# In binary fields, addition and subtraction are the same thing
galois_sub = galois_add

# Multiply two values in the Galois field
def galois_mul(x, y):
    return 0 if x*y == 0 else gexptable[glogtable[x] + glogtable[y]]

# Divide two values in the Galois field
def galois_div(x, y):
    return 0 if x == 0 else gexptable[(glogtable[x] - glogtable[y]) % two_to_the_degree_m1]

# Evaluate a polynomial at a point
def eval_poly_at(p, x):
    if x == 0:
        return p[0]
    y = 0
    logx = glogtable[x]
    for i, p_coeff in enumerate(p):
        if p_coeff:
            # Add x**i * coeff
            y ^= gexptable[(logx * i + glogtable[p_coeff]) % two_to_the_degree_m1]
    return y


# Given p+1 y values and x values with no errors, recovers the original
# p+1 degree polynomial.
# Lagrange interpolation works roughly in the following way.
# 1. Suppose you have a set of points, eg. x = [1, 2, 3], y = [2, 5, 10]
# 2. For each x, generate a polynomial which equals its corresponding
#    y coordinate at that point and 0 at all other points provided.
# 3. Add these polynomials together.

def lagrange_interp(pieces, xs):
    # Generate master numerator polynomial, eg. (x - x1) * (x - x2) * ... * (x - xn)
    root = [1]
    for x in xs:
        logx = glogtable[x]
        root.insert(0, 0)
        for j in range(len(root)-1):
            if root[j+1] and x:
                root[j] ^= gexptable[glogtable[root[j+1]] + logx]
    #print(root)
    assert len(root) == len(pieces) + 1
    # print(root)
    # Generate per-value numerator polynomials, eg. for x=x2,
    # (x - x1) * (x - x3) * ... * (x - xn), by dividing the master
    # polynomial back by each x coordinate
    nums = []
    for x in xs:
        output = [0] * (len(root) - 2) + [1]
        logx = glogtable[x]
        for j in range(len(root) - 2, 0, -1):
            if output[j] and x:
                output[j-1] = root[j] ^ gexptable[glogtable[output[j]] + logx]
            else:
                output[j-1] = root[j]
        assert len(output) == len(pieces)
        nums.append(output)
    #print(nums)
    # Generate denominators by evaluating numerator polys at each x
    denoms = [eval_poly_at(nums[i], xs[i]) for i in range(len(xs))]
    # Generate output polynomial, which is the sum of the per-value numerator
    # polynomials rescaled to have the right y values
    b = [0 for p in pieces]
    for i in range(len(xs)):
        log_yslice = glogtable[pieces[i]] - glogtable[denoms[i]] + two_to_the_degree_m1
        for j in range(len(pieces)):
            if nums[i][j] and pieces[i]:
                b[j] ^= gexptable[glogtable[nums[i][j]] + log_yslice]
    return b


a = 124
b = 8932
c = 12415

assert galois_mul(galois_add(a, b), c) == galois_add(galois_mul(a, c), galois_mul(b, c))
