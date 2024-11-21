from py_ecc import optimized_bls12_381 as b
import random

# sum(p_i * c_i)
# Note: actual implementations should use a faster algorithm, see
# https://github.com/ethereum/research/blob/master/fast_linear_combinations/multicombs.py
def linear_combination(points, coeffs, zero=b.Z1):
    o = zero
    for point, coeff in zip(points, coeffs):
        o = b.add(o, b.multiply(point, coeff))
    return o

# Helper for generating full setup
def generate_one_sided_setup(length, secret, generator=b.G1):
    o = [generator]
    for i in range(1, length):
        o.append(b.multiply(o[-1], secret))
    return o

# Generate a trusted setup with the given secret
def generate_setup(G1_length, G2_length, secret):
    return (
        generate_one_sided_setup(G1_length, secret, b.G1),
        generate_one_sided_setup(G2_length, secret, b.G2),
    )

# Verifies the integrity of a setup
def verify_setup(setup):
    G1_setup, G2_setup = setup
    G1_random_coeffs = [random.randrange(2**40) for _ in range(len(G1_setup) - 1)]
    G1_lower = linear_combination(G1_setup[:-1], G1_random_coeffs, b.Z1)
    G1_upper = linear_combination(G1_setup[1:], G1_random_coeffs, b.Z1)
    G2_random_coeffs = [random.randrange(2**40) for _ in range(len(G2_setup) - 1)]
    G2_lower = linear_combination(G2_setup[:-1], G2_random_coeffs, b.Z2)
    G2_upper = linear_combination(G2_setup[1:], G2_random_coeffs, b.Z2)
    return (
        G1_setup[0] == b.G1 and 
        G2_setup[0] == b.G2 and 
        b.pairing(G2_lower, G1_upper) == b.pairing(G2_upper, G1_lower)
    )

# Helper for extending a setup
def extend_one_sided_setup(setup, secret):
    return [
        b.multiply(point, pow(secret, i, b.curve_order))
        for i, point in enumerate(setup)
    ]

# Extends an existing setup with a new secret
def extend_setup(setup, secret):
    return (
        extend_one_sided_setup(setup[0], secret),
        extend_one_sided_setup(setup[1], secret),
    )

# Generate a proof that can be used to check that a new setup actually is
# an extension of a previous one
def get_extension_proof(previous_setup, secret):
    return (
        # G1*s from the previous setup
        b.G1 if previous_setup is None else previous_setup[0][1],
        # G2*t (the new secret being introduced)
        b.multiply(b.G2, secret)
    )

# Given a sequence of proofs from N participants, verifies that each of those
# participants actually participated
def verify_extension_proofs(final_setup, proofs):
    G1_points = [proof[0] for proof in proofs] + [final_setup[0][1]]
    G2_points = [proof[1] for proof in proofs]
    return G1_points[0] == b.G1 and all(
        b.pairing(G2_points[i], G1_points[i]) == b.pairing(b.G2, G1_points[i+1])
        for i in range(len(G2_points))
    )

# General-purpose FFT method
def fft(vals, modulus, domain, add, multiply, neg):
    if len(vals) == 1:
        return vals
    L = fft(vals[::2], modulus, domain[::2], add, multiply, neg)
    R = fft(vals[1::2], modulus, domain[::2], add, multiply, neg)
    o = [0 for i in vals]
    for i, (x, y) in enumerate(zip(L, R)):
        y_times_root = multiply(y, domain[i])
        o[i] = add(x, y_times_root)
        o[i+len(L)] = add(x, neg(y_times_root))
    return o

# Generate a w such that w**length = 1
def generate_root_of_unity(length):
    # 7 is a primitive root
    assert (b.curve_order - 1) % length == 0
    return pow(7, (b.curve_order - 1) // length, b.curve_order)

# Converts a G1 or G2 portion of a setup into the Lagrange basis
def lagrangify(setup):
    root_of_unity = generate_root_of_unity(len(setup))
    assert pow(root_of_unity, len(setup), b.curve_order) == 1
    domain = [pow(root_of_unity, i, b.curve_order) for i in range(len(setup))]
    fft_output = fft(setup, b.curve_order, domain, b.add, b.multiply, b.neg)
    inv_length = pow(len(setup), b.curve_order - 2, b.curve_order)
    return [b.multiply(fft_output[-i], inv_length) for i in range(len(fft_output))]
