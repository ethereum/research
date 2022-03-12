from trusted_setup import generate_setup, extend_setup, \
    verify_setup, get_extension_proof, verify_extension_proofs, \
    lagrangify, generate_root_of_unity, linear_combination, b

def evaluate_polynomial(coeffs, x):
    o = 0
    coeff_power = 1
    for coeff in coeffs:
        o += coeff * coeff_power
        coeff_power = (coeff_power * x) % b.curve_order
    return o % b.curve_order

def check_setup_eq(L1, L2):
    G1_1, G2_1 = L1
    G1_2, G2_2 = L2
    return (
        all(b.eq(item1, item2) for item1, item2 in zip(G1_1, G1_2)) and
        all(b.eq(item1, item2) for item1, item2 in zip(G2_1, G2_2))
    )

def test():
    S1 = generate_setup(8, 4, 42)
    S2 = extend_setup(S1, 69)
    S3 = extend_setup(S2, 1337)
    assert check_setup_eq(S3, generate_setup(8, 4, 42 * 69 * 1337))
    print("Setup extension works")
    assert verify_setup(S3)
    print("Basic setup verification works")
    proof1 = get_extension_proof(None, 42)
    proof2 = get_extension_proof(S1, 69)
    proof3 = get_extension_proof(S2, 1337)
    assert verify_extension_proofs(S3, [proof1, proof2, proof3])
    print("Extension proof verification works")
    L3 = lagrangify(S3[0])
    poly = [6, 28, 31, 85, 30, 71, 79, 58]
    root_of_unity = generate_root_of_unity(8)
    domain = [pow(root_of_unity, i, b.curve_order) for i in range(8)]
    evaluations = [evaluate_polynomial(poly, value) for value in domain]
    assert b.eq(linear_combination(S3[0], poly), linear_combination(L3, evaluations))
    print("Lagrange basis conversion works")

if __name__ == '__main__':
    test()
