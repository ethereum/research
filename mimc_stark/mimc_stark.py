from merkle_tree import merkelize, mk_branch, verify_branch, blake
from compression import compress_fri, decompress_fri, compress_branches, decompress_branches, bin_length
from ecpoly import PrimeField
from better_lagrange import lagrange_interp_4, lagrange_interp_2
import time
from fft import fft
from fri import prove_low_degree, verify_low_degree_proof
from utils import get_power_cycle, get_pseudorandom_indices

modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
nonresidue = 7

spot_check_security_factor = 240

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
    root_of_unity = pow(7, (modulus-1)//precision, modulus)

    # Root of unity such that x^skips=1
    skips = precision // steps
    subroot = pow(root_of_unity, skips)

    # Powers of the root of unity, our computational trace will be
    # along the sequence of roots of unity
    xs = get_power_cycle(subroot, modulus)
    last_step_position = xs[steps-1]

    # Generate the computational trace
    constants = []
    values = [inp]
    k = 1
    for i in range(steps-1):
        values.append((values[-1]**3 + (k ^ 1)) % modulus)
        constants.append(k ^ 1)
        k = (k * 9) & ((1 << 256) - 1)
    constants.append(0)
    output = values[-1]
    print('Done generating computational trace')

    # Interpolate the computational trace into a polynomial
    values_polynomial = fft(values, modulus, subroot, inv=True)
    constants_polynomial = fft(constants, modulus, subroot, inv=True)
    print('Converted computational steps and constants into a polynomial')

    # Create the composed polynomial such that
    # C(P(x), P(rx), K(x)) = P(rx) - P(x)**3 - K(x)
    term1 = multiply_base(values_polynomial, subroot)
    p_evaluations = fft(values_polynomial, modulus, root_of_unity)
    term2 = fft([pow(x, 3, modulus) for x in p_evaluations], modulus, root_of_unity, inv=True)[:len(values_polynomial) * 3 - 2]
    c_of_values = f.sub_polys(f.sub_polys(term1, term2), constants_polynomial)
    print('Computed C(P, K) polynomial')

    # Compute D(x) = C(P(x), P(rx), K(x)) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    d = divide_by_xnm1(f.mul_polys(c_of_values,
                                   [-last_step_position, 1]),
                       steps)
    # Consistency check
    assert (f.eval_poly_at(d, 90833) * 
            (pow(90833, steps, modulus) - 1) *
            f.inv(f.eval_poly_at([-last_step_position, 1], 90833)) -
            f.eval_poly_at(c_of_values, 90833)) % modulus == 0
    print('Computed D polynomial')

    # Compute interpolant of ((1, input), (x_atlast_step, output))
    interpolant = lagrange_interp_2([inp, output], [1, last_step_position], modulus)
    quotient = f.mul_polys([-1, 1], [-last_step_position, 1])
    b = f.div_polys(f.sub_polys(values_polynomial, interpolant), quotient)
    # Consistency check
    assert f.eval_poly_at(f.add_polys(f.mul_polys(b, quotient), interpolant), 7045) == f.eval_poly_at(values_polynomial, 7045)
    print('Computed B polynomial')

    # Evaluate B, D and K across the entire subgroup
    d_evaluations = fft(d, modulus, root_of_unity)
    k_evaluations = fft(constants_polynomial, modulus, root_of_unity)
    b_evaluations = fft(b, modulus, root_of_unity)
    print('Evaluated low-degree extension of B, D and K')

    # Compute their Merkle roots
    p_mtree = merkelize(p_evaluations)
    d_mtree = merkelize(d_evaluations)
    k_mtree = merkelize(k_evaluations)
    b_mtree = merkelize(b_evaluations)
    print('Computed hash root')

    # Based on the hashes of P, D and B, we select a random linear combination
    # of P * x^steps, P, B * x^steps, B and D, and prove the low-degreeness of that,
    # instead of proving the low-degreeness of P, B and D separately
    k1 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x01'), 'big')
    k2 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x02'), 'big')
    k3 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x03'), 'big')
    k4 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x04'), 'big')

    lincomb = f.add_polys(f.add_polys(d,
                                      f.mul_by_const(values_polynomial, k1) + f.mul_by_const(values_polynomial, k2)),
                          f.mul_by_const(b, k3) + [0, 0] + f.mul_by_const(b, k4) + [0,0])
    l_evaluations = fft(lincomb, modulus, root_of_unity)
    l_mtree = merkelize(l_evaluations)

    print('Computed random linear combination')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates
    branches = []
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_pseudorandom_indices(l_mtree[1], precision - skips, samples)
    for pos in positions:
        branches.append(mk_branch(p_mtree, pos))
        branches.append(mk_branch(p_mtree, pos + skips))
        branches.append(mk_branch(d_mtree, pos))
        branches.append(mk_branch(k_mtree, pos))
        branches.append(mk_branch(b_mtree, pos))
        branches.append(mk_branch(l_mtree, pos))
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [p_mtree[1],
         d_mtree[1],
         k_mtree[1],
         b_mtree[1],
         l_mtree[1],
         branches,
         prove_low_degree(lincomb, root_of_unity, l_evaluations, steps * 2, modulus)]
    print("STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a STARK
def verify_mimc_proof(inp, logsteps, logprecision, output, proof):
    p_root, d_root, k_root, b_root, l_root, branches, fri_proof = proof
    start_time = time.time()

    steps = 2**logsteps
    precision = 2**logprecision

    # Get (steps)th root of unity
    root_of_unity = pow(7, (modulus-1)//precision, modulus)
    skips = precision // steps

    # Verifies the low-degree proofs
    assert verify_low_degree_proof(l_root, root_of_unity, fri_proof, steps * 2, modulus)

    # Performs the spot checks
    k1 = int.from_bytes(blake(p_root + d_root + b_root + b'\x01'), 'big')
    k2 = int.from_bytes(blake(p_root + d_root + b_root + b'\x02'), 'big')
    k3 = int.from_bytes(blake(p_root + d_root + b_root + b'\x03'), 'big')
    k4 = int.from_bytes(blake(p_root + d_root + b_root + b'\x04'), 'big')
    samples = spot_check_security_factor // (logprecision - logsteps)
    positions = get_pseudorandom_indices(l_root, precision - skips, samples)
    last_step_position = pow(root_of_unity, (steps - 1) * skips, modulus)
    for i, pos in enumerate(positions):
        # Check C(P(x)) = Z(x) * D(x)
        x = pow(root_of_unity, pos, modulus)
        x_to_the_steps = pow(x, steps, modulus)
        p_of_x = verify_branch(p_root, pos, branches[i*6])
        p_of_rx = verify_branch(p_root, pos+skips, branches[i*6 + 1])
        d_of_x = verify_branch(d_root, pos, branches[i*6 + 2])
        k_of_x = verify_branch(k_root, pos, branches[i*6 + 3])
        b_of_x = verify_branch(b_root, pos, branches[i*6 + 4])
        l_of_x = verify_branch(l_root, pos, branches[i*6 + 5])
        zvalue = f.div(pow(x, steps, modulus) - 1,
                       x - last_step_position)
        assert (p_of_rx - p_of_x ** 3 - k_of_x - zvalue * d_of_x) % modulus == 0
        interpolant = lagrange_interp_2([inp, output], [1, last_step_position], modulus)
        quotient = f.mul_polys([-1, 1], [-last_step_position, 1])
        assert (p_of_x - b_of_x * f.eval_poly_at(quotient, x) -
                f.eval_poly_at(interpolant, x)) % modulus == 0
        assert (l_of_x - d_of_x - 
                k1 * p_of_x - k2 * p_of_x * x_to_the_steps -
                k3 * b_of_x - k4 * b_of_x * x_to_the_steps) % modulus == 0

    print('Verified %d consistency checks' % (spot_check_security_factor // (logprecision - logsteps)))
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    print('Note: this does not include verifying the Merkle root of the constants tree')
    print('This can be done by every client once as a precomputation')
    return True
