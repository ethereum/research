from circom_tools import *
from Crypto.Hash import keccak

def verify_proof(setup, group_order, vk, proof):
    (
        A_pt, B_pt, C_pt, Z_pt, T1_pt, T2_pt, T3_pt, W_z_pt, W_zw_pt,
        A_ev, B_ev, C_ev, S1_ev, S2_ev, Z_shifted_ev
    ) = proof

    Ql_pt, Qr_pt, Qm_pt, Qo_pt, Qc_pt, S1_pt, S2_pt, S3_pt, X2 = (
        vk["Ql"], vk["Qr"], vk["Qm"], vk["Qo"], vk["Qc"],
        vk["S1"], vk["S2"], vk["S3"], vk["X_2"]
    )

    # Compute challenges (should be same as those computed by prover)

    buf = serialize_point(A_pt) + serialize_point(B_pt) + serialize_point(C_pt)

    beta = binhash_to_f_inner(keccak256(buf))
    gamma = binhash_to_f_inner(keccak256(keccak256(buf)))

    alpha = binhash_to_f_inner(keccak256(serialize_point(Z_pt)))

    buf2 = serialize_point(T1_pt)+serialize_point(T2_pt)+serialize_point(T3_pt)
    zed = binhash_to_f_inner(keccak256(buf))

    buf3 = b''.join([
        x.n.to_bytes(32, 'big') for x in
        (A_ev, B_ev, C_ev, S1_ev, S2_ev, Z_shifted_ev)
    ])
    v = binhash_to_f_inner(keccak256(buf3))

    #print(
    #    "Verif challenges: \n\nbeta: {}\ngamma: {}\nalpha: {}\nzed: {}\nv: {}"
    #    .format(beta, gamma, alpha, zed, v)
    #)

    # Does not need to be standardized, only needs to be unpredictable
    u = binhash_to_f_inner(keccak256(buf + buf2 + buf3))

    ZH_ev = zed ** group_order - 1

    root_of_unity = get_root_of_unity(group_order)

    L1_ev = (
        (zed ** group_order - 1) /
        (group_order * (zed - 1))
    )

    R_pt = ec_lincomb([
        (Qm_pt, (A_ev * B_ev).n),
        (Ql_pt, A_ev.n),
        (Qr_pt, B_ev.n), 
        (Qo_pt, C_ev.n), 
        (Qc_pt, 1),
        (Z_pt, ((
            (A_ev + beta * zed + gamma) *
            (B_ev + beta * 2 * zed + gamma) *
            (C_ev + beta * 3 * zed + gamma)
        ) * alpha).n),
        (S3_pt, ((
            -(A_ev + beta * S1_ev + gamma) * 
            (B_ev + beta * S2_ev + gamma) *
            beta
        ) * alpha * Z_shifted_ev).n),
        (b.G1, ((
            -(A_ev + beta * S1_ev + gamma) * 
            (B_ev + beta * S2_ev + gamma) *
            (C_ev + gamma)
        ) * alpha * Z_shifted_ev).n),
        (Z_pt, L1_ev * alpha ** 2),
        (b.G1, -L1_ev * alpha ** 2),
        (T1_pt, -ZH_ev),
        (T2_pt, -ZH_ev * zed**group_order),
        (T3_pt, -ZH_ev * zed**(group_order*2)),
    ])

    print('verifier R_pt', R_pt)

    # Constant term of r
    r0 = (
        -L1_ev * alpha ** 2 - (
            alpha *
            (A_ev + beta * S1_ev + gamma) *
            (B_ev + beta * S2_ev + gamma) *
            (C_ev + gamma) *
            Z_shifted_ev
        )
    )

    D_pt = ec_lincomb([
        (Qm_pt, (A_ev * B_ev).n),
        (Ql_pt, A_ev.n),
        (Qr_pt, B_ev.n), 
        (Qo_pt, C_ev.n), 
        (Qc_pt, 1),
        (Z_pt, ((
            (A_ev + beta * zed + gamma) *
            (B_ev + beta * 2 * zed + gamma) *
            (C_ev + beta * 3 * zed + gamma)
        ) * alpha + L1_ev * alpha**2 + u).n),
        (S3_pt, (
            -(A_ev + beta * S1_ev + gamma) * 
            (B_ev + beta * S2_ev + gamma) * 
            alpha * beta * Z_shifted_ev
        ).n),
        (T1_pt, -ZH_ev),
        (T2_pt, -ZH_ev * zed**group_order),
        (T3_pt, -ZH_ev * zed**(group_order*2)),
    ])

    assert ec_lincomb([(R_pt, 1), (b.G1, -r0), (Z_pt, u)]) == D_pt

    F_pt = ec_lincomb([
        (D_pt, 1),
        (A_pt, v),
        (B_pt, v**2),
        (C_pt, v**3),
        (S1_pt, v**4),
        (S2_pt, v**5),
    ])
    E_pt = b.multiply(b.G1, (
        -r0 + v * A_ev + v**2 * B_ev + v**3 * C_ev +
        v**4 * S1_ev + v**5 * S2_ev + u * Z_shifted_ev
    ).n)

    assert b.pairing(
        b.G2,
        ec_lincomb([
            (R_pt, 1),
            (A_pt, v),
            (b.G1, -v * A_ev),
            (B_pt, v**2),
            (b.G1, -v**2 * B_ev),
            (C_pt, v**3),
            (b.G1, -v**3 * C_ev),
            (S1_pt, v**4),
            (b.G1, -v**4 * S1_ev),
            (S2_pt, v**5),
            (b.G1, -v**5 * S2_ev),
        ])
    ) == b.pairing(
        b.add(X2, b.multiply(b.G2, (-zed).n)),
        W_z_pt
    )
    print("done check 1")

    assert b.pairing(
        b.G2,
        ec_lincomb([
            (Z_pt, 1),
            (b.G1, -Z_shifted_ev)
        ])
    ) == b.pairing(
        b.add(X2, b.multiply(b.G2, (-zed*root_of_unity).n)),
        W_zw_pt
    )
    print("done check 2")

    assert b.pairing(X2, ec_lincomb([
        (W_z_pt, 1),
        (W_zw_pt, u)
    ])) == b.pairing(b.G2, ec_lincomb([
        (W_z_pt, zed),
        (W_zw_pt, u * zed * root_of_unity),
        (F_pt, 1),
        (E_pt, -1)
    ]))

    print("done combined check!")
    return True
