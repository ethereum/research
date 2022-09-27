from circom_tools import *
from Crypto.Hash import keccak

def keccak256(x):
    return keccak.new(digest_bits=256).update(x).digest()

def serialize_point(pt):
    return pt[0].n.to_bytes(32, 'big') + pt[1].n.to_bytes(32, 'big')

def binhash_to_f_inner(h):
    return f_inner(int.from_bytes(h, 'big'))

def prove_from_witness(setup, group_order, eqs, var_assignments):
    eqs = [eq_to_coeffs(eq) if isinstance(eq, str) else eq for eq in eqs]
    if None not in var_assignments:
        var_assignments[None] = 0
    variables = [v for (v, c) in eqs]
    # Compute wire assignments
    A = [0] * group_order
    B = [0] * group_order
    C = [0] * group_order
    for i, (in_L, in_R, out) in enumerate(variables):
        A[i] = var_assignments[in_L]
        B[i] = var_assignments[in_R]
        C[i] = var_assignments[out]
    A_pt = evaluations_to_point(setup, group_order, A)
    B_pt = evaluations_to_point(setup, group_order, B)
    C_pt = evaluations_to_point(setup, group_order, C)

    buf = serialize_point(A_pt) + serialize_point(B_pt) + serialize_point(C_pt)

    beta = binhash_to_f_inner(keccak256(buf))
    gamma = binhash_to_f_inner(keccak256(keccak256(buf)))

    # Compute the accumulator polynomial for the permutation arguments
    print(variables)
    S1, S2, S3 = make_s_polynomials(group_order, variables)
    Z = [f_inner(1)]
    roots_of_unity = get_roots_of_unity(group_order)
    for i in range(group_order):
        Z.append(
            Z[-1] *
            (A[i] + beta * roots_of_unity[i] + gamma) *
            (B[i] + beta * 2 * roots_of_unity[i] + gamma) *
            (C[i] + beta * 3 * roots_of_unity[i] + gamma) /
            (A[i] + beta * S1[i] + gamma) /
            (B[i] + beta * S2[i] + gamma) /
            (C[i] + beta * S3[i] + gamma)
        )
    assert Z.pop().n == 1
    Z_pt = evaluations_to_point(setup, group_order, Z)
    alpha = binhash_to_f_inner(keccak256(serialize_point(Z_pt)))
    print("Permutation accumulator polynomial successfully generated")

    # Compute the quotient polynomial

    # List of roots of unity at 4x fineness
    quarter_roots = get_roots_of_unity(group_order * 4)

    # This value could be anything, it just needs to be unpredictable. Lets us
    # have evaluation forms at cosets to avoid zero evaluations, so we can
    # divide polys without the 0/0 issue
    fft_offset = binhash_to_f_inner(keccak256(keccak256(serialize_point(Z_pt))))

    def fft_expand(values):
        if hasattr(values[0], 'n'):
            values = [x.n for x in values]
        x_powers = fft(values, b.curve_order, roots_of_unity[1], inv=True)
        x_powers = [
            (fft_offset**i * x).n for i, x in enumerate(x_powers)
        ] + [0] * (group_order * 3)
        return [
            f_inner(x) for x in fft(x_powers, b.curve_order, quarter_roots[1])
        ]

    def expanded_evaluations_to_coeffs(evals):
        shifted_coeffs = fft(
            [i.n for i in evals], b.curve_order, quarter_roots[1], inv=True
        )
        inv_offset = (1 / fft_offset).n
        return [
            (pow(inv_offset, i, b.curve_order) * v) % b.curve_order for
            (i, v) in enumerate(shifted_coeffs)
        ]
        

    A_big = fft_expand(A)
    B_big = fft_expand(B)
    C_big = fft_expand(C)
    # Z_H = X^N - 1, also in evaluation form in the coset
    ZH_big = [
        ((f_inner(r) * fft_offset) ** group_order - 1)
        for r in quarter_roots
    ]

    QL, QR, QM, QO, QC = make_gate_polynomials(group_order, eqs)

    QL_big, QR_big, QM_big, QO_big, QC_big = \
        (fft_expand(x) for x in (QL, QR, QM, QO, QC))

    for i in range(group_order):
        assert (A[i] * QL[i] + B[i] * QR[i] + A[i] * B[i] * QM[i] + C[i] * QO[i] + QC[i]) % b.curve_order == 0

    QUOT_part_1_big = [(
        A_big[i] * QL_big[i] +
        B_big[i] * QR_big[i] +
        A_big[i] * B_big[i] * QM_big[i] +
        C_big[i] * QO_big[i] + 
        QC_big[i]
    ) / ZH_big[i] for i in range(group_order * 4)]
    
    assert (
        expanded_evaluations_to_coeffs(QUOT_part_1_big)[-2*group_order:] ==
        [0] * (2 * group_order)
    )
    print("Generated part 1 of the quotient polynomial")

    Z_big = fft_expand(Z)
    Z_shifted_big = Z_big[4:] + Z_big[:4]
    S1_big = fft_expand(S1)
    S2_big = fft_expand(S2)
    S3_big = fft_expand(S3)

    for i in range(group_order):
        assert (
            (A[i] + beta * roots_of_unity[i] + gamma) * 
            (B[i] + beta * 2 * roots_of_unity[i] + gamma) * 
            (C[i] + beta * 3 * roots_of_unity[i] + gamma)
        ) * Z[i] - (
            (A[i] + beta * S1[i] + gamma) * 
            (B[i] + beta * S2[i] + gamma) * 
            (C[i] + beta * S3[i] + gamma)
        ) * Z[(i+1) % group_order] == 0

    QUOT_part_2_big = [(
        (
            (A_big[i] + beta * fft_offset * quarter_roots[i] + gamma) *
            (B_big[i] + beta * 2 * fft_offset * quarter_roots[i] + gamma) *
            (C_big[i] + beta * 3 * fft_offset * quarter_roots[i] + gamma)
        ) * alpha * Z_big[i] - (
            (A_big[i] + beta * S1_big[i] + gamma) *
            (B_big[i] + beta * S2_big[i] + gamma) *
            (C_big[i] + beta * S3_big[i] + gamma)
        ) * alpha * Z_shifted_big[i]
    ) / ZH_big[i] for i in range(group_order * 4)]

    assert (
        expanded_evaluations_to_coeffs(QUOT_part_2_big)[-group_order:] ==
        [0] * group_order
    )
    print("Generated part 2 of the quotient polynomial")

    L1_big = fft_expand([1,0,0,0,0,0,0,0])
    
    QUOT_part_3_big = [(
        (Z_big[i] - 1) * L1_big[i] * alpha**2
    ) / ZH_big[i] for i in range(group_order * 4)]

    assert (
        expanded_evaluations_to_coeffs(QUOT_part_3_big)[-3*group_order:] ==
        [0] * (3 * group_order)
    )
    print("Generated part 3 of the quotient polynomial")

    all_coeffs = expanded_evaluations_to_coeffs([
        QUOT_part_1_big[i] + QUOT_part_2_big[i] + QUOT_part_3_big[i]
        for i in range(4 * group_order)
    ])

    (T1, T2, T3) = (
        fft(all_coeffs[group_order*i : group_order*(i+1)],
            b.curve_order, roots_of_unity[1])
        for i in range(3)
    )
    T1_pt = evaluations_to_point(setup, group_order, T1)
    T2_pt = evaluations_to_point(setup, group_order, T2)
    T3_pt = evaluations_to_point(setup, group_order, T3)
    print("Generated T1, T2, T3 polynomials")

    # TODO: finish round 4 and round 5
