from merkle_tree import merkelize, mk_branch, verify_branch, blake
from compression import compress_fri, decompress_fri, compress_branches, decompress_branches, bin_length
from ecpoly import PrimeField
import time

modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
nonresidue = 7
quartic_roots_of_unity = [1,
                          pow(7, (modulus-1)//4, modulus),
                          pow(7, (modulus-1)//2, modulus),
                          pow(7, (modulus-1)*3//4, modulus)]

spot_check_security_factor = 240

# Treat a polynomial as a bivariate polynomial g(x, y) and
# evaluate it as such. Invariant: eval_as_bivariate(p, x, x**4) = eval(p, x)
def eval_as_bivariate(p, x, y):
    o = 0
    ypow = 1
    xpows = [pow(x, i, modulus) for i in range(4)]
    for i in range(0, len(p), 4):
        for j in range(4):
            o += xpows[j] * ypow * p[i+j]
        ypow = (ypow * y) % modulus
    return o % modulus

# Get the set of powers of R, until but not including when the powers
# loop back to 1
def get_power_cycle(r):
    o = [1, r]
    while o[-1] != 1:
        o.append((o[-1] * r) % modulus)
    return o[:-1]

# Extract pseudorandom indices from entropy
def get_indices(seed, modulus, count):
    assert modulus < 2**24
    data = seed
    while len(data) < 4 * count:
        data += blake(data[-32:])
    return [int.from_bytes(data[i: i+4], 'big') % modulus for i in range(0, count * 4, 4)]

# Generate an FRI proof
def prove_low_degree(poly, root_of_unity, values, maxdeg_plus_1):
    print('Proving %d values are degree <= %d' % (len(values), maxdeg_plus_1))

    # If the degree we are checking for is less than or equal to 32,
    # use the polynomial directly as a proof
    if maxdeg_plus_1 <= 32:
        print('Produced FRI proof')
        return [[x.to_bytes(32, 'big') for x in values]]

    # Calculate the set of x coordinates
    xs = get_power_cycle(root_of_unity)

    # Put the values into a Merkle tree. This is the root that the
    # proof will be checked against
    m = merkelize(values)

    # Select a pseudo-random x coordinate
    special_x = int.from_bytes(m[1], 'big') % modulus

    # Calculate the "column" (see https://vitalik.ca/general/2017/11/22/starks_part_2.html)
    # at that x coordinate
    column = [eval_as_bivariate(poly, special_x, xs[i]) for i in range(0, len(xs), 4)]
    m2 = merkelize(column)

    # Pseudo-randomly select y indices to sample
    ys = get_indices(m2[1], len(column), 40)

    # Compute the Merkle branches for the values in the polynomial and the column
    branches = []
    for y in ys:
        branches.append([mk_branch(m2, y)] + [mk_branch(m, y + (len(xs) // 4) * j) for j in range(4)])

    #for j in range(4):
    #    assert values[ys[0] + len(xs) // 4 * j] == eval_as_bivariate(poly, xs[ys[0] + len(xs) // 4 * j], xs[ys[0] * 4])
    #assert column[ys[0]] == eval_as_bivariate(poly, special_x, xs[ys[0] * 4])

    # This component of the proof
    o = [m2[1], branches]

    # In the next iteration of the proof, we'll work with smaller roots of unity
    sub_xs = [xs[i] for i in range(0, len(xs), 4)]

    # Interpolate the polynomial for the column
    ypoly = f.lagrange_interp(column[:len(sub_xs)], sub_xs)

    # Recurse...
    return [o] + prove_low_degree(ypoly, pow(root_of_unity, 4, modulus), column, maxdeg_plus_1 // 4)

# Verify an FRI proof
def verify_low_degree_proof(merkle_root, root_of_unity, proof, maxdeg_plus_1):

    # Calculate which root of unity we're working with
    testval = root_of_unity
    roudeg = 1
    while testval != 1:
        roudeg *= 2
        testval = (testval * testval) % modulus

    # Verify the recursive components of the proof
    for prf in proof[:-1]:
        root2, branches = prf
        print('Verifying degree <= %d' % maxdeg_plus_1)

        # Calculate the pseudo-random x coordinate
        special_x = int.from_bytes(merkle_root, 'big') % modulus

        # Calculate the pseudo-randomly sampled y indices
        ys = get_indices(root2, roudeg // 4, 40)


        # Verify for each selected y coordinate that the four points from the polynomial
        # and the one point from the column that are on that y coordinate are on a
        # deg < 4 polynomial
        for i, y in enumerate(ys):
            # The five x coordinates we are checking
            x1 = pow(root_of_unity, y, modulus)
            eckses = [special_x] + [(quartic_roots_of_unity[j] * x1) % modulus for j in range(4)]

            # The values from the polynomial
            row = [verify_branch(merkle_root, y + (roudeg // 4) * j, prf) for j, prf in zip(range(4), branches[i][1:])]

            # Verify proof and recover the column value
            values = [verify_branch(root2, y, branches[i][0])] + row

            # Lagrange interpolate and check deg is < 4
            p = f.lagrange_interp(values, eckses)
            assert p[4] == 0

        # Update constants to check the next proof
        merkle_root = root2
        root_of_unity = pow(root_of_unity, 4, modulus)
        maxdeg_plus_1 //= 4
        roudeg //= 4

    # Verify the direct components of the proof
    data = [int.from_bytes(x, 'big') for x in proof[-1]]
    print('Verifying degree <= %d' % maxdeg_plus_1)
    assert maxdeg_plus_1 <= 32

    # Check the Merkle root matches up
    mtree = merkelize(data)
    assert mtree[1] == merkle_root

    # Check its degree
    xs = get_power_cycle(root_of_unity)
    poly = f.lagrange_interp(data[:maxdeg_plus_1], xs[:maxdeg_plus_1])
    for x, datum in zip(xs[maxdeg_plus_1:], data[maxdeg_plus_1:]):
        assert f.eval_poly_at(poly, x) == datum

    print('FRI proof verified')
    return True

# Pure FRI tests
poly = list(range(512))
root_of_unity = pow(7, (modulus-1)//1024, modulus)
evaluations = [f.eval_poly_at(poly, pow(root_of_unity, i, modulus)) for i in range(1024)]
proof = prove_low_degree(poly, root_of_unity, evaluations, 512)
print("Approx proof length: %d" % bin_length(compress_fri(proof)))
assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 512)

# Compute a MIMC permutation for 2**logsteps steps, using round constants
# from the multiplicative subgroup of size 2**logprecision
def mimc(inp, logsteps, logprecision):        
    start_time = time.time()
    steps = 2**logsteps
    precision = 2**logprecision
    # Get (steps)th root of unity
    root = pow(7, (modulus-1)//precision, modulus)
    xs = get_power_cycle(root)
    for i in range(steps-1):
        inp = (inp**3 + xs[i]) % modulus
    print("MIMC computed in %.4f sec" % (time.time() - start_time))
    return inp

# Generate a STARK for a MIMC calculation
def mk_mimc_proof(inp, logsteps, logprecision):
    start_time = time.time()
    assert logsteps < logprecision <= 32
    steps = 2**logsteps
    precision = 2**logprecision

    # Get (steps)th root of unity
    root = pow(7, (modulus-1)//precision, modulus)
    # Powers of the root of unity, our computational trace will be
    # along the sequence of roots of unity
    xs = get_power_cycle(root)

    # Generate the computational trace
    values = [inp]
    for i in range(steps-1):
        values.append((values[-1]**3 + xs[i]) % modulus)
    print('Done generating computational trace')

    # Interpolate the computational trace into a polynomial
    values_polynomial = f.lagrange_interp(values, xs[:len(values)])
    print('Computed polynomial')

    #for x, v in zip(xs, values):
    #    assert f.eval_poly_at(values_polynomial, x) == v

    # Create the composed polynomial such that
    # C(P(x), P(rx)) = P(rx) - P(x)**3 - x
    term1 = f.compose_polys(values_polynomial, [0, root])
    term2 = f.mul_polys(f.mul_polys(values_polynomial, values_polynomial),
                        values_polynomial)
    c_of_values = f.sub_polys(f.sub_polys(term1, term2), [0, 1])

    #for i in range(steps-1):
    #    assert f.eval_poly_at(c_of_values, xs[i]) == 0
    #print('C(P(x)) check passed')

    # Compute the Z(x) polynomial that is 0 along the trace
    z = f.zpoly(xs[:steps-1])
    print('Computed Z polynomial')

    # Compute D(x) = C(P(x)) / Z(x)
    d = f.div_polys(c_of_values, z)
    assert f.mul_polys(d, z) == c_of_values
    print('Computed and checked D polynomial')

    # Evaluate P and D across the entire subgroup
    p_evaluations = [f.eval_poly_at(values_polynomial, x) for x in xs]
    d_evaluations = [f.eval_poly_at(d, x) for x in xs]
    print('Evaluated P and D')

    # Compute their Merkle roots
    p_mtree = merkelize(p_evaluations)
    d_mtree = merkelize(d_evaluations)
    print('Computed hash root')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates
    branches = []
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_indices(blake(p_mtree[1] + d_mtree[1]), len(xs) - 1, samples)
    for pos in positions:
        branches.append(mk_branch(p_mtree, pos))
        branches.append(mk_branch(p_mtree, pos + 1))
        branches.append(mk_branch(d_mtree, pos))
    print('Computed %d spot checks' % samples)

    while len(d) < steps * 2:
        d += [0]

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [p_mtree[1],
         d_mtree[1],
         branches,
         prove_low_degree(values_polynomial, root, p_evaluations, steps),
         prove_low_degree(d, root, d_evaluations, steps * 2)]
    print("STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a STARK
def verify_mimc_proof(inp, logsteps, logprecision, output, zvalues, proof):
    p_root, d_root, branches, p_proof, d_proof = proof
    start_time = time.time()

    steps = 2**logsteps
    precision = 2**logprecision

    # Get (steps)th root of unity
    root_of_unity = pow(7, (modulus-1)//precision, modulus)

    # Verifies the low-degree proofs
    assert verify_low_degree_proof(p_root, root_of_unity, p_proof, steps)
    assert verify_low_degree_proof(d_root, root_of_unity, d_proof, steps * 2)

    # Performs the spot checks
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_indices(blake(p_root + d_root), len(xs) - 1, samples)
    for i, pos in enumerate(positions):

        # Check C(P(x)) = Z(x) * D(x)
        x = pow(root_of_unity, pos, modulus)
        p_of_x = verify_branch(p_root, pos, branches[i*3])
        p_of_rx = verify_branch(p_root, pos+1, branches[i*3 + 1])
        d_of_x = verify_branch(d_root, pos, branches[i*3 + 2])
        assert (p_of_rx - p_of_x ** 3 - x - zvalues[pos] * d_of_x) % modulus == 0

    print('Verified %d consistency checks' % (spot_check_security_factor // (logprecision - logsteps)))
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    return True

INPUT = 3
LOGSTEPS = 8
LOGPRECISION = 11

# Full STARK test
proof = mk_mimc_proof(INPUT, LOGSTEPS, LOGPRECISION)
L1 = bin_length(compress_branches(proof[2]))
L2 = bin_length(compress_fri(proof[3]))
L3 = bin_length(compress_fri(proof[4]))
print("Approx proof length: %d (branches), %d (FRI proof 1), %d (FRI proof 2), %d (total)" % (L1, L2, L3, L1 + L2 + L3))
xs = get_power_cycle(pow(7, (modulus-1)//2**LOGPRECISION, modulus))
zpoly = f.zpoly(xs[:2**LOGSTEPS-1])
zpoly_vals = [f.eval_poly_at(zpoly, x) for x in xs]
assert verify_mimc_proof(3, LOGSTEPS, LOGPRECISION, mimc(3, LOGSTEPS, LOGPRECISION), zpoly_vals, proof)
