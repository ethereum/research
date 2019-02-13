from permuted_tree import merkelize, mk_branch, verify_branch, blake, mk_multi_branch, verify_multi_branch
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
    G2 = f.exp(7, (modulus-1)//precision)

    # Root of unity such that x^steps=1
    skips = precision // steps
    G1 = f.exp(G2, skips)

    # Powers of the higher-order root of unity
    xs = get_power_cycle(G2, modulus)
    last_step_position = xs[(steps-1)*extension_factor]

    # Generate the computational trace
    computational_trace = [inp]
    for i in range(steps-1):
        computational_trace.append(
            (computational_trace[-1]**3 + round_constants[i % len(round_constants)]) % modulus
        )
    output = computational_trace[-1]
    print('Done generating computational trace')

    # Interpolate the computational trace into a polynomial P, with each step
    # along a successive power of G1
    computational_trace_polynomial = fft(computational_trace, modulus, G1, inv=True)
    p_evaluations = fft(computational_trace_polynomial, modulus, G2)
    print('Converted computational steps into a polynomial and low-degree extended it')

    skips2 = steps // len(round_constants)
    constants_mini_polynomial = fft(round_constants, modulus, f.exp(G1, skips2), inv=True)
    constants_polynomial = [0 if i % skips2 else constants_mini_polynomial[i//skips2] for i in range(steps)]
    constants_mini_extension = fft(constants_mini_polynomial, modulus, f.exp(G2, skips2))
    print('Converted round constants into a polynomial and low-degree extended it')

    # Create the composed polynomial such that
    # C(P(x), P(g1*x), K(x)) = P(g1*x) - P(x)**3 - K(x)
    c_of_p_evaluations = [(p_evaluations[(i+extension_factor)%precision] -
                              f.exp(p_evaluations[i], 3) -
                              constants_mini_extension[i % len(constants_mini_extension)])
                          % modulus for i in range(precision)]
    print('Computed C(P, K) polynomial')

    # Compute D(x) = C(P(x), P(g1*x), K(x)) / Z(x)
    # Z(x) = (x^steps - 1) / (x - x_atlast_step)
    z_num_evaluations = [xs[(i * steps) % precision] - 1 for i in range(precision)]
    z_num_inv = f.multi_inv(z_num_evaluations)
    z_den_evaluations = [xs[i] - last_step_position for i in range(precision)]
    d_evaluations = [cp * zd * zni % modulus for cp, zd, zni in zip(c_of_p_evaluations, z_den_evaluations, z_num_inv)]
    print('Computed D polynomial')

    # Compute interpolant of ((1, input), (x_atlast_step, output))
    interpolant = f.lagrange_interp_2([1, last_step_position], [inp, output])
    i_evaluations = [f.eval_poly_at(interpolant, x) for x in xs]

    zeropoly2 = f.mul_polys([-1, 1], [-last_step_position, 1])
    inv_z2_evaluations = f.multi_inv([f.eval_poly_at(zeropoly2, x) for x in xs])

    b_evaluations = [((p - i) * invq) % modulus for p, i, invq in zip(p_evaluations, i_evaluations, inv_z2_evaluations)]
    print('Computed B polynomial')

    # Compute their Merkle root
    mtree = merkelize([pval.to_bytes(32, 'big') +
                       dval.to_bytes(32, 'big') +
                       bval.to_bytes(32, 'big') for
                       pval, dval, bval in zip(p_evaluations, d_evaluations, b_evaluations)])
    print('Computed hash root')

    # Based on the hashes of P, D and B, we select a random linear combination
    # of P * x^steps, P, B * x^steps, B and D, and prove the low-degreeness of that,
    # instead of proving the low-degreeness of P, B and D separately
    k1 = int.from_bytes(blake(mtree[1] + b'\x01'), 'big')
    k2 = int.from_bytes(blake(mtree[1] + b'\x02'), 'big')
    k3 = int.from_bytes(blake(mtree[1] + b'\x03'), 'big')
    k4 = int.from_bytes(blake(mtree[1] + b'\x04'), 'big')

    # Compute the linear combination. We don't even both calculating it in
    # coefficient form; we just compute the evaluations
    G2_to_the_steps = f.exp(G2, steps)
    powers = [1]
    for i in range(1, precision):
        powers.append(powers[-1] * G2_to_the_steps % modulus)

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
    augmented_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    #for pos in positions:
    #    branches.append(mk_branch(mtree, pos))
    #    branches.append(mk_branch(mtree, (pos + skips) % precision))
    #    branches.append(mk_branch(l_mtree, pos))
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    o = [mtree[1],
         l_mtree[1],
         mk_multi_branch(mtree, augmented_positions),
         mk_multi_branch(l_mtree, positions),
         prove_low_degree(l_evaluations, G2, steps * 2, modulus, exclude_multiples_of=extension_factor)]
    print("STARK computed in %.4f sec" % (time.time() - start_time))
    return o

# Verifies a STARK
def verify_mimc_proof(inp, steps, round_constants, output, proof):
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    start_time = time.time()
    assert steps <= 2**32 // extension_factor
    assert is_a_power_of_2(steps) and is_a_power_of_2(len(round_constants))
    assert len(round_constants) < steps

    precision = steps * extension_factor

    # Get (steps)th root of unity
    G2 = f.exp(7, (modulus-1)//precision)
    skips = precision // steps

    # Gets the polynomial representing the round constants
    skips2 = steps // len(round_constants)
    constants_mini_polynomial = fft(round_constants, modulus, f.exp(G2, extension_factor * skips2), inv=True)

    # Verifies the low-degree proofs
    assert verify_low_degree_proof(l_root, G2, fri_proof, steps * 2, modulus, exclude_multiples_of=extension_factor)

    # Performs the spot checks
    k1 = int.from_bytes(blake(m_root + b'\x01'), 'big')
    k2 = int.from_bytes(blake(m_root + b'\x02'), 'big')
    k3 = int.from_bytes(blake(m_root + b'\x03'), 'big')
    k4 = int.from_bytes(blake(m_root + b'\x04'), 'big')
    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    augmented_positions = sum([[x, (x + skips) % precision] for x in positions], [])
    last_step_position = f.exp(G2, (steps - 1) * skips)
    main_branch_leaves = verify_multi_branch(m_root, augmented_positions, main_branches)
    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)
    for i, pos in enumerate(positions):
        x = f.exp(G2, pos)
        x_to_the_steps = f.exp(x, steps)
        mbranch1 = main_branch_leaves[i*2]
        mbranch2 = main_branch_leaves[i*2+1]
        l_of_x = int.from_bytes(linear_comb_branch_leaves[i], 'big')

        p_of_x = int.from_bytes(mbranch1[:32], 'big')
        p_of_g1x = int.from_bytes(mbranch2[:32], 'big')
        d_of_x = int.from_bytes(mbranch1[32:64], 'big')
        b_of_x = int.from_bytes(mbranch1[64:], 'big')

        zvalue = f.div(f.exp(x, steps) - 1,
                       x - last_step_position)
        k_of_x = f.eval_poly_at(constants_mini_polynomial, f.exp(x, skips2))

        # Check transition constraints C(P(x)) = Z(x) * D(x)
        assert (p_of_g1x - p_of_x ** 3 - k_of_x - zvalue * d_of_x) % modulus == 0

        # Check boundary constraints B(x) * Q(x) + I(x) = P(x)
        interpolant = f.lagrange_interp_2([1, last_step_position], [inp, output])
        zeropoly2 = f.mul_polys([-1, 1], [-last_step_position, 1])
        assert (p_of_x - b_of_x * f.eval_poly_at(zeropoly2, x) -
                f.eval_poly_at(interpolant, x)) % modulus == 0

        # Check correctness of the linear combination
        assert (l_of_x - d_of_x - 
                k1 * p_of_x - k2 * p_of_x * x_to_the_steps -
                k3 * b_of_x - k4 * b_of_x * x_to_the_steps) % modulus == 0

    print('Verified %d consistency checks' % spot_check_security_factor)
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    return True
