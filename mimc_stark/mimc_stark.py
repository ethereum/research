from merkle_tree import merkelize, mk_branch, verify_branch, blake
from compression import compress_fri, decompress_fri, compress_branches, decompress_branches, bin_length
from poly_utils import PrimeField
import time
from fft import fft
from fri import prove_low_degree, verify_low_degree_proof
from utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2

modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
nonresidue = 7

spot_check_security_factor = 80
extension_factor = 8

# Compute a MIMC permutation for some number of steps
def mimc(inp, steps, round_constants):
    start_time = time.time()
    for i in range(steps-1):
        inp = (inp**3 + round_constants[i % len(round_constants)]) % modulus
    print("MIMC computed in %.4f sec" % (time.time() - start_time))
    return inp

# Generate a STARK for a MIMC calculation
def mk_mimc_proof(inp, steps, round_constants):
    start_time = time.time()
    # Some constraints to make our job easier
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps) and is_a_power_of_2(len(round_constants))
    assert len(round_constants) < steps

    precision = steps * extension_factor

    # Root of unity such that x^precision=1
    root_of_unity = f.exp(7, (modulus-1)//precision)

    # Root of unity such that x^skips=1
    skips = precision // steps
    subroot = f.exp(root_of_unity, skips)

    # Powers of the root of unity, our computational trace will be
    # along the sequence of sub-roots
    xs = get_power_cycle(root_of_unity, modulus)
    last_step_position = xs[(steps-1)*extension_factor]

    # Generate the computational trace
    values = [inp]
    for i in range(steps-1):
        values.append((values[-1]**3 + round_constants[i % len(round_constants)]) % modulus)
    output = values[-1]
    print('Done generating computational trace')

    # Interpolate the computational trace into a polynomial
    values_polynomial = fft(values, modulus, subroot, inv=True)
    p_evaluations = fft(values_polynomial, modulus, root_of_unity)
    print('Converted computational steps into a polynomial and low-degree extended it')

    skips2 = steps // len(round_constants)
    constants_mini_polynomial = fft(round_constants, modulus, f.exp(subroot, skips2), inv=True)
    constants_polynomial = [0 if i % skips2 else constants_mini_polynomial[i//skips2] for i in range(steps)]
    constants_mini_extension = fft(constants_mini_polynomial, modulus, f.exp(root_of_unity, skips2))
    print('Converted round constants into a polynomial and low-degree extended it')

    # Create the composed polynomial such that
    # C(P(x), P(rx), K(x)) = P(rx) - P(x)**3 - K(x)
    c_of_p_evaluations = [(p_evaluations[(i+extension_factor)%precision] -
                              f.exp(p_evaluations[i], 3) -
                              constants_mini_extension[i % len(constants_mini_extension)])
                          % modulus for i in range(precision)]
    print('Computed C(P, K) polynomial')

    # Compute D(x) = C(P(x), P(rx), K(x)) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    z_num_evaluations = [xs[(i * steps) % precision] - 1 for i in range(precision)]
    z_num_inv = f.multi_inv(z_num_evaluations)
    z_den_evaluations = [xs[i] - last_step_position for i in range(precision)]
    d_evaluations = [cp * zd * zni % modulus for cp, zd, zni in zip(c_of_p_evaluations, z_den_evaluations, z_num_inv)]
    print('Computed D polynomial')

    # Compute interpolant of ((1, input), (x_atlast_step, output))
    interpolant = f.lagrange_interp_2([1, last_step_position], [inp, output])
    i_evaluations = [f.eval_poly_at(interpolant, x) for x in xs]

    quotient = f.mul_polys([-1, 1], [-last_step_position, 1])
    inv_q_evaluations = f.multi_inv([f.eval_poly_at(quotient, x) for x in xs])

    b_evaluations = [((p - i) * invq) % modulus for p, i, invq in zip(p_evaluations, i_evaluations, inv_q_evaluations)]
    print('Computed B polynomial')

    # Compute their Merkle roots
    p_mtree = merkelize(p_evaluations)
    d_mtree = merkelize(d_evaluations)
    b_mtree = merkelize(b_evaluations)
    print('Computed hash root')

    # Based on the hashes of P, D and B, we select a random linear combination
    # of P * x^steps, P, B * x^steps, B and D, and prove the low-degreeness of that,
    # instead of proving the low-degreeness of P, B and D separately
    k1 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x01'), 'big')
    k2 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x02'), 'big')
    k3 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x03'), 'big')
    k4 = int.from_bytes(blake(p_mtree[1] + d_mtree[1] + b_mtree[1] + b'\x04'), 'big')

    # Compute the linear combination. We don't even both calculating it in
    # coefficient form; we just compute the evaluations
    root_of_unity_to_the_steps = f.exp(root_of_unity, steps)
    powers = [1]
    for i in range(1, precision):
        powers.append(powers[-1] * root_of_unity_to_the_steps % modulus)

    l_evaluations = [(d_evaluations[i] +
                      p_evaluations[i] * k1 + p_evaluations[i] * k2 * powers[i] +
                      b_evaluations[i] * k3 + b_evaluations[i] * powers[i] * k4) % modulus
                      for i in range(precision)]

    l_mtree = merkelize(l_evaluations)
    print('Computed random linear combination')

    # Do some spot checks of the Merkle tree at pseudo-random coordinates, excluding
    # multiples of `extension_factor`
    branches = []
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples,
                                         exclude_multiples_of=extension_factor)
    for pos in positions:
        branches.append(mk_branch(p_mtree, pos))
        branches.append(mk_branch(p_mtree, (pos + skips) % precision))
        branches.append(mk_branch(d_mtree, pos))
        branches.append(mk_branch(b_mtree, pos))
        branches.append(mk_branch(l_mtree, pos))
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [p_mtree[1],
         d_mtree[1],
         b_mtree[1],
         l_mtree[1],
         branches,
         prove_low_degree(l_evaluations, root_of_unity, steps * 2, modulus, exclude_multiples_of=extension_factor)]
    print("STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a STARK
def verify_mimc_proof(inp, steps, round_constants, output, proof):
    p_root, d_root, b_root, l_root, branches, fri_proof = proof
    start_time = time.time()
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps) and is_a_power_of_2(len(round_constants))
    assert len(round_constants) < steps

    precision = steps * extension_factor

    # Get (steps)th root of unity
    root_of_unity = f.exp(7, (modulus-1)//precision)
    skips = precision // steps

    # Gets the polynomial representing the round constants
    skips2 = steps // len(round_constants)
    constants_mini_polynomial = fft(round_constants, modulus, f.exp(root_of_unity, extension_factor * skips2), inv=True)

    # Verifies the low-degree proofs
    assert verify_low_degree_proof(l_root, root_of_unity, fri_proof, steps * 2, modulus, exclude_multiples_of=extension_factor)

    # Performs the spot checks
    k1 = int.from_bytes(blake(p_root + d_root + b_root + b'\x01'), 'big')
    k2 = int.from_bytes(blake(p_root + d_root + b_root + b'\x02'), 'big')
    k3 = int.from_bytes(blake(p_root + d_root + b_root + b'\x03'), 'big')
    k4 = int.from_bytes(blake(p_root + d_root + b_root + b'\x04'), 'big')
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    last_step_position = f.exp(root_of_unity, (steps - 1) * skips)
    for i, pos in enumerate(positions):
        x = f.exp(root_of_unity, pos)
        x_to_the_steps = f.exp(x, steps)
        p_of_x = verify_branch(p_root, pos, branches[i*5])
        p_of_rx = verify_branch(p_root, (pos+skips)%precision, branches[i*5 + 1])
        d_of_x = verify_branch(d_root, pos, branches[i*5 + 2])
        b_of_x = verify_branch(b_root, pos, branches[i*5 + 3])
        l_of_x = verify_branch(l_root, pos, branches[i*5 + 4])

        zvalue = f.div(f.exp(x, steps) - 1,
                       x - last_step_position)
        k_of_x = f.eval_poly_at(constants_mini_polynomial, f.exp(x, skips2))

        # Check transition constraints C(P(x)) = Z(x) * D(x)
        assert (p_of_rx - p_of_x ** 3 - k_of_x - zvalue * d_of_x) % modulus == 0
        interpolant = f.lagrange_interp_2([1, last_step_position], [inp, output])
        quotient = f.mul_polys([-1, 1], [-last_step_position, 1])

        # Check boundary constraints B(x) * Q(x) + I(x) = P(x)
        assert (p_of_x - b_of_x * f.eval_poly_at(quotient, x) -
                f.eval_poly_at(interpolant, x)) % modulus == 0

        # Check correctness of the linear combination
        assert (l_of_x - d_of_x - 
                k1 * p_of_x - k2 * p_of_x * x_to_the_steps -
                k3 * b_of_x - k4 * b_of_x * x_to_the_steps) % modulus == 0

    print('Verified %d consistency checks' % spot_check_security_factor)
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    return True
