from circom_tools import *
from Crypto.Hash import keccak

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

    L1_big = fft_expand([1] + [0] * (group_order - 1))
    
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

    buf2 = serialize_point(T1_pt)+serialize_point(T2_pt)+serialize_point(T3_pt)
    zed = binhash_to_f_inner(keccak256(buf))

    def evaluate_at_point(values, x):
        if not hasattr(values[0], 'n'):
            values = [f_inner(x) for x in values]
        order = len(values)
        roots_of_unity = get_roots_of_unity(order)
        return (
            (f_inner(x)**order - 1) / order *
            sum([
                values[i] * roots_of_unity[i] / (x - roots_of_unity[i])
                for i in range(order)
            ])
        )

    assert (
        evaluate_at_point(T1, fft_offset) +
        evaluate_at_point(T2, fft_offset) * fft_offset**group_order +
        evaluate_at_point(T3, fft_offset) * fft_offset**(group_order*2)
    ) == (
        QUOT_part_1_big[0] + QUOT_part_2_big[0] + QUOT_part_3_big[0]
    )

    A_ev = evaluate_at_point(A, zed)
    B_ev = evaluate_at_point(B, zed)
    C_ev = evaluate_at_point(C, zed)
    S1_ev = evaluate_at_point(S1, zed)
    S2_ev = evaluate_at_point(S2, zed)
    Z_shifted_ev = evaluate_at_point(Z, zed * roots_of_unity[1])

    L1_ev = evaluate_at_point([1] + [0] * (group_order - 1), zed)
    ZH_ev = zed ** group_order - 1

    T1_big = fft_expand(T1)
    T2_big = fft_expand(T2)
    T3_big = fft_expand(T3)

    R_big = [(
        A_ev * QL_big[i] +
        B_ev * QR_big[i] +
        A_ev * B_ev * QM_big[i] +
        C_ev * QO_big[i] +
        QC_big[i]
    ) + (
        (A_ev + beta * zed + gamma) *
        (B_ev + beta * 2 * zed + gamma) *
        (C_ev + beta * 3 * zed + gamma)
    ) * alpha * Z_big[i] - (
        (A_ev + beta * S1_ev + gamma) * 
        (B_ev + beta * S2_ev + gamma) *
        (C_ev + beta * S3_big[i] + gamma)
    ) * alpha * Z_shifted_ev + (
        (Z_big[i] - 1) * L1_ev
    ) * alpha**2 - (
        T1_big[i] +
        zed ** group_order * T2_big[i] +
        zed ** (group_order * 2) * T3_big[i]
    ) * ZH_ev for i in range(4 * group_order)]

    R_coeffs = expanded_evaluations_to_coeffs(R_big)
    assert R_coeffs[group_order:] == [0] * (group_order * 3)
    R = fft(R_coeffs[:group_order], b.curve_order, roots_of_unity[1])

    print('R_pt', evaluations_to_point(setup, group_order, R))

    assert evaluate_at_point(R, zed) == 0

    print("Generated linearization polynomial R")

    buf3 = b''.join([
        x.n.to_bytes(32, 'big') for x in
        (A_ev, B_ev, C_ev, S1_ev, S2_ev, Z_shifted_ev)
    ])
    v = binhash_to_f_inner(keccak256(buf3))

    W_z_big = [(
        R_big[i] +
        v * (A_big[i] - A_ev) +
        v**2 * (B_big[i] - B_ev) +
        v**3 * (C_big[i] - C_ev) +
        v**4 * (S1_big[i] - S1_ev) +
        v**5 * (S2_big[i] - S2_ev)
    ) / (fft_offset * quarter_roots[i] - zed) for i in range(group_order * 4)]

    W_z_coeffs = expanded_evaluations_to_coeffs(W_z_big)
    assert W_z_coeffs[group_order:] == [0] * (group_order * 3)
    W_z = fft(W_z_coeffs[:group_order], b.curve_order, roots_of_unity[1])
    W_z_pt = evaluations_to_point(setup, group_order, W_z)

    W_zw_big = [
        (Z_big[i] - Z_shifted_ev) /
        (fft_offset * quarter_roots[i] - zed * roots_of_unity[1])
    for i in range(group_order * 4)]

    W_zw_coeffs = expanded_evaluations_to_coeffs(W_zw_big)
    assert W_zw_coeffs[group_order:] == [0] * (group_order * 3)
    W_zw = fft(W_zw_coeffs[:group_order], b.curve_order, roots_of_unity[1])
    W_zw_pt = evaluations_to_point(setup, group_order, W_zw)

    print("Generated final quotient witness polynomials")
    #print(
    #    "Prover challenges: \n\nbeta: {}\ngamma: {}\nalpha: {}\nzed: {}\nv: {}"
    #    .format(beta, gamma, alpha, zed, v)
    #)
    return (
        A_pt, B_pt, C_pt, Z_pt, T1_pt, T2_pt, T3_pt, W_z_pt, W_zw_pt,
        A_ev, B_ev, C_ev, S1_ev, S2_ev, Z_shifted_ev
    )
