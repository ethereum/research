from merkle_tree import merkelize, mk_branch, verify_branch, blake
from compression import compress_fri, decompress_fri, compress_branches, decompress_branches, bin_length
from ecpoly import PrimeField
from better_lagrange import lagrange_interp_4
import time
from fft import fft

modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
nonresidue = 7
quartic_roots_of_unity = [1,
                          pow(7, (modulus-1)//4, modulus),
                          pow(7, (modulus-1)//2, modulus),
                          pow(7, (modulus-1)*3//4, modulus)]

spot_check_security_factor = 240

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
    # We calculate the column by Lagrange-interpolating the row, and not
    # directly, as this is more efficient
    column = []
    for i in range(len(xs)//4):
        x_poly = lagrange_interp_4(
            [values[i+len(values)*j//4] for j in range(4)],
            [xs[i+len(xs)*j//4] for j in range(4)],
            modulus
        )
        column.append(f.eval_poly_at(x_poly, special_x))
    m2 = merkelize(column)

    # Pseudo-randomly select y indices to sample
    ys = get_indices(m2[1], len(column), 40)

    # Compute the Merkle branches for the values in the polynomial and the column
    branches = []
    for y in ys:
        branches.append([mk_branch(m2, y)] + [mk_branch(m, y + (len(xs) // 4) * j) for j in range(4)])

    # This component of the proof
    o = [m2[1], branches]

    # In the next iteration of the proof, we'll work with smaller roots of unity
    sub_xs = [xs[i] for i in range(0, len(xs), 4)]

    # Interpolate the polynomial for the column
    ypoly = fft(column[:len(sub_xs)], modulus,
                pow(root_of_unity, 4, modulus), inv=True)

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
        # and the one point from the column that are on that y coordinate are on the same
        # deg < 4 polynomial
        for i, y in enumerate(ys):
            # The x coordinates from the polynomial
            x1 = pow(root_of_unity, y, modulus)
            xcoords = [(quartic_roots_of_unity[j] * x1) % modulus for j in range(4)]

            # The values from the polynomial
            row = [verify_branch(merkle_root, y + (roudeg // 4) * j, prf) for j, prf in zip(range(4), branches[i][1:])]

            # Verify proof and recover the column value
            values = [verify_branch(root2, y, branches[i][0])] + row

            # Lagrange interpolate and check deg is < 4
            p = lagrange_interp_4(row, xcoords, modulus)
            assert f.eval_poly_at(p, special_x) == verify_branch(root2, y, branches[i][0])

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

    # Check the degree of the data
    poly = fft(data, modulus, root_of_unity, inv=True)
    for i in range(maxdeg_plus_1, len(poly)):
        assert poly[i] == 0

    print('FRI proof verified')
    return True

# Pure FRI tests
poly = list(range(512))
root_of_unity = pow(7, (modulus-1)//1024, modulus)
evaluations = fft(poly, modulus, root_of_unity)
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
    subroot = pow(7, (modulus-1)//steps, modulus)
    # We use powers of 9 mod 2^256 as the ith round constant for the moment
    k = 1
    for i in range(steps-1):
        inp = (inp**3 + (k ^ 1)) % modulus
        k = (k * 9) & ((1 << 256) - 1)
    print("MIMC computed in %.4f sec" % (time.time() - start_time))
    return inp

# Convert a polynomial P(x) into a polynomial Q(x) = P(fac * x)
def multiply_base(poly, fac):
    o = []
    r = 1
    for p in poly:
        o.append(p * r % modulus)
        r = r * fac % modulus
    return o

# Divides a polynomial by x^n-1
def divide_by_xnm1(poly, n):
    if len(poly) <= n:
        return []
    return f.add_polys(poly[n:], divide_by_xnm1(poly[n:], n))

# Generate a STARK for a MIMC calculation
def mk_mimc_proof(inp, logsteps, logprecision):
    start_time = time.time()
    assert logsteps < logprecision <= 32
    steps = 2**logsteps
    precision = 2**logprecision

    # Root of unity such that x^precision=1
    root = pow(7, (modulus-1)//precision, modulus)

    # Root of unity such that x^skips=1
    skips = precision // steps
    subroot = pow(root, skips)

    # Powers of the root of unity, our computational trace will be
    # along the sequence of roots of unity
    xs = get_power_cycle(subroot)

    # Generate the computational trace
    constants = []
    values = [inp]
    k = 1
    for i in range(steps-1):
        values.append((values[-1]**3 + (k ^ 1)) % modulus)
        constants.append(k ^ 1)
        k = (k * 9) & ((1 << 256) - 1)
    constants.append(0)
    print('Done generating computational trace')

    # Interpolate the computational trace into a polynomial
    values_polynomial = fft(values, modulus, subroot, inv=True)
    constants_polynomial = fft(constants, modulus, subroot, inv=True)
    print('Converted computational steps and constants into a polynomial')

    # Create the composed polynomial such that
    # C(P(x), P(rx), K(x)) = P(rx) - P(x)**3 - K(x)
    term1 = multiply_base(values_polynomial, subroot)
    p_evaluations = fft(values_polynomial, modulus, root)
    term2 = fft([pow(x, 3, modulus) for x in p_evaluations], modulus, root, inv=True)[:len(values_polynomial) * 3 - 2]
    c_of_values = f.sub_polys(f.sub_polys(term1, term2), constants_polynomial)
    print('Computed C(P, K) polynomial')

    # Compute D(x) = C(P(x), P(rx), K(x)) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    d = divide_by_xnm1(f.mul_polys(c_of_values,
                                   [modulus-xs[steps-1], 1]),
                       steps)
    # assert f.mul_polys(d, z) == c_of_values
    print('Computed D polynomial')

    # Evaluate D and K across the entire subgroup
    d_evaluations = fft(d, modulus, root)
    k_evaluations = fft(constants_polynomial, modulus, root)
    print('Evaluated P, D and K')

    # Compute their Merkle roots
    p_mtree = merkelize(p_evaluations)
    d_mtree = merkelize(d_evaluations)
    k_mtree = merkelize(k_evaluations)
    print('Computed hash root')

    # Based on the hashes of P and D, we select a random linear combination
    # of P * x^steps and D, and prove the low-degreeness of that, instead of proving
    # the low-degreeness of P and D separately
    k = int.from_bytes(blake(p_mtree[1] + d_mtree[1]), 'big')

    lincomb = f.add_polys(d, f.mul_by_const([0] * steps + values_polynomial, k))
    l_evaluations = fft(lincomb, modulus, root)
    l_mtree = merkelize(l_evaluations)

    print('Computed random linear combination')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates
    branches = []
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_indices(l_mtree[1], precision - skips, samples)
    for pos in positions:
        branches.append(mk_branch(p_mtree, pos))
        branches.append(mk_branch(p_mtree, pos + skips))
        branches.append(mk_branch(d_mtree, pos))
        branches.append(mk_branch(k_mtree, pos))
        branches.append(mk_branch(l_mtree, pos))
    print('Computed %d spot checks' % samples)


    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [p_mtree[1],
         d_mtree[1],
         k_mtree[1],
         l_mtree[1],
         branches,
         prove_low_degree(lincomb, root, l_evaluations, steps * 2)]
    print("STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a STARK
def verify_mimc_proof(inp, logsteps, logprecision, output, proof):
    p_root, d_root, k_root, l_root, branches, fri_proof = proof
    start_time = time.time()

    steps = 2**logsteps
    precision = 2**logprecision

    # Get (steps)th root of unity
    root_of_unity = pow(7, (modulus-1)//precision, modulus)
    skips = precision // steps

    # Verifies the low-degree proofs
    assert verify_low_degree_proof(l_root, root_of_unity, fri_proof, steps * 2)

    # Performs the spot checks
    k = int.from_bytes(blake(p_root + d_root), 'big')
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_indices(l_root, precision - skips, samples)
    for i, pos in enumerate(positions):

        # Check C(P(x)) = Z(x) * D(x)
        x = pow(root_of_unity, pos, modulus)
        p_of_x = verify_branch(p_root, pos, branches[i*5])
        p_of_rx = verify_branch(p_root, pos+skips, branches[i*5 + 1])
        d_of_x = verify_branch(d_root, pos, branches[i*5 + 2])
        k_of_x = verify_branch(k_root, pos, branches[i*5 + 3])
        l_of_x = verify_branch(l_root, pos, branches[i*5 + 4])
        zvalue = f.div(pow(x, steps, modulus) - 1,
                       x - pow(root_of_unity, (steps - 1) * skips, modulus))
        assert (p_of_rx - p_of_x ** 3 - k_of_x - zvalue * d_of_x) % modulus == 0
        assert (l_of_x - d_of_x - k * p_of_x * pow(x, steps, modulus)) % modulus == 0

    print('Verified %d consistency checks' % (spot_check_security_factor // (logprecision - logsteps)))
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    print('Note: this does not include verifying the Merkle root of the constants tree')
    print('This can be done by every client once as a precomputation')
    return True

INPUT = 3
LOGSTEPS = 17
LOGPRECISION = 20

# Full STARK test
proof = mk_mimc_proof(INPUT, LOGSTEPS, LOGPRECISION)
p_root, d_root, k_root, l_root, branches, fri_proof = proof
L1 = bin_length(compress_branches(branches))
L2 = bin_length(compress_fri(fri_proof))
print("Approx proof length: %d (branches), %d (FRI proof), %d (total)" % (L1, L2, L1 + L2))
root_of_unity = pow(7, (modulus-1)//2**LOGPRECISION, modulus)
subroot = pow(7, (modulus-1)//2**LOGSTEPS, modulus)
skips = 2**(LOGPRECISION - LOGSTEPS)
assert verify_mimc_proof(3, LOGSTEPS, LOGPRECISION, mimc(3, LOGSTEPS, LOGPRECISION), proof)
