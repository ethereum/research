from utils import *

def verify_proof(setup, group_order, vk, proof, public=[], optimized=True):
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
    zed = binhash_to_f_inner(keccak256(buf2))

    buf3 = b''.join([
        serialize_int(x) for x in
        (A_ev, B_ev, C_ev, S1_ev, S2_ev, Z_shifted_ev)
    ])
    v = binhash_to_f_inner(keccak256(buf3))

    # Does not need to be standardized, only needs to be unpredictable
    u = binhash_to_f_inner(keccak256(buf + buf2 + buf3))

    ZH_ev = zed ** group_order - 1

    root_of_unity = get_root_of_unity(group_order)

    L1_ev = (
        (zed ** group_order - 1) /
        (group_order * (zed - 1))
    )

    PI_ev = barycentric_eval_at_point(
        [f_inner(-x) for x in public] +
        [f_inner(0) for _ in range(group_order - len(public))],
        zed
    )

    if not optimized:
        # Basic, easier-to-understand version of what's going on

        # Recover the commitment to the linearization polynomial R,
        # exactly the same as what was created by the prover
        R_pt = ec_lincomb([
            (Qm_pt, A_ev * B_ev),
            (Ql_pt, A_ev),
            (Qr_pt, B_ev), 
            (Qo_pt, C_ev), 
            (b.G1, PI_ev),
            (Qc_pt, 1),
            (Z_pt, (
                (A_ev + beta * zed + gamma) *
                (B_ev + beta * 2 * zed + gamma) *
                (C_ev + beta * 3 * zed + gamma) *
                alpha
            )),
            (S3_pt, (
                -(A_ev + beta * S1_ev + gamma) * 
                (B_ev + beta * S2_ev + gamma) *
                beta *
                alpha * Z_shifted_ev
            )),
            (b.G1, (
                -(A_ev + beta * S1_ev + gamma) * 
                (B_ev + beta * S2_ev + gamma) *
                (C_ev + gamma) *
                alpha * Z_shifted_ev
            )),
            (Z_pt, L1_ev * alpha ** 2),
            (b.G1, -L1_ev * alpha ** 2),
            (T1_pt, -ZH_ev),
            (T2_pt, -ZH_ev * zed**group_order),
            (T3_pt, -ZH_ev * zed**(group_order*2)),
        ])
    
        print('verifier R_pt', R_pt)
    
        # Verify that R(z) = 0 and the prover-provided evaluations
        # A(z), B(z), C(z), S1(z), S2(z) are all correct
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
            b.add(X2, ec_mul(b.G2, -zed)),
            W_z_pt
        )
        print("done check 1")
    
        # Verify that the provided value of Z(zed*w) is correct
        assert b.pairing(
            b.G2,
            ec_lincomb([
                (Z_pt, 1),
                (b.G1, -Z_shifted_ev)
            ])
        ) == b.pairing(
            b.add(X2, ec_mul(b.G2, -zed * root_of_unity)),
            W_zw_pt
        )
        print("done check 2")
        return True
    
    else:
        # More optimized version that tries hard to minimize pairings and
        # elliptic curve multiplications, but at the cost of being harder
        # to understand and mixing together a lot of the computations to
        # efficiently batch them
        
        # Compute the constant term of R. This is not literally the degree-0
        # term of the R polynomial; rather, it's the portion of R that can
        # be computed directly, without resorting to elliptic cutve commitments
        r0 = (
            PI_ev - L1_ev * alpha ** 2 - (
                alpha *
                (A_ev + beta * S1_ev + gamma) *
                (B_ev + beta * S2_ev + gamma) *
                (C_ev + gamma) *
                Z_shifted_ev
            )
        )
    
        # D = (R - r0) + u * Z
        D_pt = ec_lincomb([
            (Qm_pt, A_ev * B_ev),
            (Ql_pt, A_ev),
            (Qr_pt, B_ev), 
            (Qo_pt, C_ev), 
            (Qc_pt, 1),
            (Z_pt, (
                (A_ev + beta * zed + gamma) *
                (B_ev + beta * 2 * zed + gamma) *
                (C_ev + beta * 3 * zed + gamma) * alpha +
                L1_ev * alpha ** 2 +
                u
            )),
            (S3_pt, (
                -(A_ev + beta * S1_ev + gamma) * 
                (B_ev + beta * S2_ev + gamma) * 
                alpha * beta * Z_shifted_ev
            )),
            (T1_pt, -ZH_ev),
            (T2_pt, -ZH_ev * zed**group_order),
            (T3_pt, -ZH_ev * zed**(group_order*2)),
        ])
    
        F_pt = ec_lincomb([
            (D_pt, 1),
            (A_pt, v),
            (B_pt, v**2),
            (C_pt, v**3),
            (S1_pt, v**4),
            (S2_pt, v**5),
        ])

        E_pt = ec_mul(b.G1, (
            -r0 + v * A_ev + v**2 * B_ev + v**3 * C_ev +
            v**4 * S1_ev + v**5 * S2_ev + u * Z_shifted_ev
        ))
    
        # What's going on here is a clever re-arrangement of terms to check
        # the same equations that are being checked in the basic version,
        # but in a way that minimizes the number of EC muls and even
        # compressed the two pairings into one. The 2 pairings -> 1 pairing
        # trick is basically to replace checking
        #
        # Y1 = A * (X - a) and Y2 = B * (X - b)
        #
        # with
        #
        # Y1 + A * a = A * X
        # Y2 + B * b = B * X
        #
        # so at this point we can take a random linear combination of the two
        # checks, and verify it with only one pairing.
        assert b.pairing(X2, ec_lincomb([
            (W_z_pt, 1),
            (W_zw_pt, u)
        ])) == b.pairing(b.G2, ec_lincomb([
            (W_z_pt, zed),
            (W_zw_pt, u * zed * root_of_unity),
            (F_pt, 1),
            (E_pt, -1)
        ]))
    
        print("done combined check")
        return True
