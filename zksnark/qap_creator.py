# Polynomials are stored as arrays, where the ith element in
# the array is the ith degree coefficient

# Multiply two polynomials
def multiply_polys(a, b):
    o = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            o[i + j] += a[i] * b[j]
    return o

# Add two polynomials
def add_polys(a, b, subtract=False):
    o = [0] * max(len(a), len(b))
    for i in range(len(a)):
        o[i] += a[i]
    for i in range(len(b)):
        o[i] += b[i] * (-1 if subtract else 1) # Reuse the function structure for subtraction
    return o

def subtract_polys(a, b):
    return add_polys(a, b, subtract=True)

# Divide a/b, return quotient and remainder
def div_polys(a, b):
    o = [0] * (len(a) - len(b) + 1)
    remainder = a
    while len(remainder) >= len(b):
        leading_fac = remainder[-1] / b[-1]
        pos = len(remainder) - len(b)
        o[pos] = leading_fac
        remainder = subtract_polys(remainder, multiply_polys(b, [0] * pos + [leading_fac]))[:-1]
    return o, remainder

# Evaluate a polynomial at a point
def eval_poly(poly, x):
    return sum([poly[i] * x**i for i in range(len(poly))])

# Make a polynomial which is zero at {1, 2 ... total_pts}, except
# for `point_loc` where the value is `height`
def mk_singleton(point_loc, height, total_pts):
    fac = 1
    for i in range(1, total_pts + 1):
        if i != point_loc:
            fac *= point_loc - i
    o = [height * 1.0 / fac]
    for i in range(1, total_pts + 1):
        if i != point_loc:
            o = multiply_polys(o, [-i, 1])
    return o

# Assumes vec[0] = p(1), vec[1] = p(2), etc, tries to find p,
# expresses result as [deg 0 coeff, deg 1 coeff...]
def lagrange_interp(vec):
    o = []
    for i in range(len(vec)):
        o = add_polys(o, mk_singleton(i + 1, vec[i], len(vec)))
    for i in range(len(vec)):
        assert abs(eval_poly(o, i + 1) - vec[i] < 10**-10), \
            (o, eval_poly(o, i + 1), i+1)
    return o

def transpose(matrix):
    return list(map(list, zip(*matrix)))
    
# A, B, C = matrices of m vectors of length n, where for each
# 0 <= i < m, we want to satisfy A[i] * B[i] - C[i] = 0
def r1cs_to_qap(A, B, C):
    A, B, C = transpose(A), transpose(B), transpose(C)
    new_A = [lagrange_interp(a) for a in A]
    new_B = [lagrange_interp(b) for b in B]
    new_C = [lagrange_interp(c) for c in C]
    Z = [1]
    for i in range(1, len(A[0]) + 1):
        Z = multiply_polys(Z, [-i, 1])
    return (new_A, new_B, new_C, Z)

def create_solution_polynomials(r, new_A, new_B, new_C):
    Apoly = []
    for rval, a in zip(r, new_A):
        Apoly = add_polys(Apoly, multiply_polys([rval], a))
    Bpoly = []
    for rval, b in zip(r, new_B):
        Bpoly = add_polys(Bpoly, multiply_polys([rval], b))
    Cpoly = []
    for rval, c in zip(r, new_C):
        Cpoly = add_polys(Cpoly, multiply_polys([rval], c))
    o = subtract_polys(multiply_polys(Apoly, Bpoly), Cpoly)
    for i in range(1, len(new_A[0]) + 1):
        assert abs(eval_poly(o, i)) < 10**-10, (eval_poly(o, i), i)
    return Apoly, Bpoly, Cpoly, o

def create_divisor_polynomial(sol, Z):
    quot, rem = div_polys(sol, Z)
    for x in rem:
        assert abs(x) < 10**-10
    return quot

r = [1, 3, 35, 9, 27, 30]
A = [[0, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 1, 0],
     [5, 0, 0, 0, 0, 1]]
B = [[0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0]]
C = [[0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0]]

Ap, Bp, Cp, Z = r1cs_to_qap(A, B, C)
print 'Ap'
for x in Ap: print x
print 'Bp'
for x in Bp: print x
print 'Cp'
for x in Cp: print x
print 'Z'
print Z
Apoly, Bpoly, Cpoly, sol = create_solution_polynomials(r, Ap, Bp, Cp)
print 'Apoly'
print Apoly
print 'Bpoly'
print Bpoly
print 'Cpoly'
print Cpoly
print 'Sol'
print sol
print 'Z cofactor'
print create_divisor_polynomial(sol, Z)
