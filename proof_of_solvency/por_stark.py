# This demo is based on Vitalik's proof of solvency proposal and implemented using the STARK proof system. 
# See https://vitalik.ca/general/2022/11/19/proof_of_solvency.html
# It provides users with proofs that constrain the sum of all assets and the non-negativity of their net asset value. 
# This is a basic version of the solution.
# Most of the modules used here including fft and fri etc. came from "../mimc_stark".

# THIS IS EDUCATIONAL CODE, NOT PRODUCTION! HIRE A SECURITY AUDITOR
# WHEN BUILDING SOMETHING FOR PRODUCTION USE.

import time
import sys
sys.path.append("..")
sys.path.append("../mimc_stark")


from mimc_stark.permuted_tree import merkelize, mk_branch, verify_branch, blake, mk_multi_branch, verify_multi_branch
from mimc_stark.poly_utils import PrimeField
from mimc_stark.fft import fft
from mimc_stark.fri import prove_low_degree, verify_low_degree_proof
from utils import *
from constants import *

f = PrimeField(MODULUS)

def mk_por_proof(ids, coins, uts, data_path):
    start_time = time.time()

    # check data into field
    check_in_field(ids, MODULUS)
    check_in_field(coins, MODULUS)
    assert is_a_power_of_2(uts) and uts <= MAX_UTS, "invalid uts"

    if not is_a_power_of_2(len(ids)+1):
        ids, coins = pad(ids, coins, MAX_USER_NUM_FOR_UTS16)
    user_num = len(ids)+1
    steps = uts * user_num
    precision = steps * EXTENSION_FACTOR

    ids, coins = extend_user_data(ids, coins, uts)
    sum_trace, sum_amounts, coins = get_sum_trace(coins, uts, MODULUS)

    # get generators
    G2 = f.exp(NONRESIDUE, (MODULUS - 1) // precision)
    skips = precision // steps
    G1 = f.exp(G2, skips)
    # get x coordinates for fft
    xs = get_power_cycle(G2, MODULUS)
    last_step_position = xs[(steps - 1) * EXTENSION_FACTOR]

    # get poly and eval for sum_trace
    t_polynomial = fft(sum_trace, MODULUS, G1, inv=True)
    t_evaluations = fft(t_polynomial, MODULUS, G2)

    # get poly and eval for values of each coin
    b_evaluations = []
    for i in range(len(coins)):
        b_polynomial = fft(coins[i], MODULUS, G1, inv=True)
        b_evaluations.append(fft(b_polynomial, MODULUS, G2))
    
    del b_polynomial, t_polynomial

    id_evaluations = sum([[x] + [0]*(EXTENSION_FACTOR-1) for x in ids], [])

    # trace constraints evaluations
    tc_evaluations = []
    # constraint 1: t[uts*EXTENSION_FACTOR*i] = 0, 0<=i<=user_num-1, 
    # z1(x) = (x-xs[uts*EXTENSION_FACTOR*0])(x-xs[uts*EXTENSION_FACTOR*1])...(x-xs[uts*EXTENSION_FACTOR*(user_num-1)]) 
    # z1(x) = x^user_num - 1
    z1_evaluations = [xs[(i * user_num) % precision] - 1 for i in range(precision)]
    z1_inv = f.multi_inv(z1_evaluations)
    tc_evaluations.append([tp1 * zi1 % MODULUS for tp1, zi1 in zip(t_evaluations, z1_inv)])

    # constraint 2:  (t[i + EXTENSION_FACTOR] - 3*t[i])*(t[i + EXTENSION_FACTOR] - 3*t[i] - 1)*(t[i + EXTENSION_FACTOR] - 3*t[i] - 2) = 0, 
    # 0<=i<=steps-1 && (i mod uts*EXTENSION_FACTOR != {(uts-2)*EXTENSION_FACTOR,(uts-1)*EXTENSION_FACTOR}, i in range(precision)
    # z2(x) = (x^steps -1)/((x^user_num -  G2^((uts-2)*EXTENSION_FACTOR*user_num))(x^user_num - G2^((uts-1)*EXTENSION_FACTOR*user_num)))
    c2_num_evaluations = [((t_evaluations[(i + EXTENSION_FACTOR) % precision] - 3 * t_evaluations[i]) * 
                      (t_evaluations[(i + EXTENSION_FACTOR) % precision] - 3 * t_evaluations[i] - 1) *
                      (t_evaluations[(i + EXTENSION_FACTOR) % precision] - 3 * t_evaluations[i] - 2)) % MODULUS for i in range(precision)]
    z2_num_evaluations = [(xs[(i * steps) % precision] - 1) % MODULUS for i in range(precision)]
    z2_num_inv = f.multi_inv(z2_num_evaluations)
    z2_den_evaluations = [(((xs[(i * user_num) % precision] - xs[(uts-2) * EXTENSION_FACTOR * user_num])) * 
                           ((xs[(i * user_num) % precision] - xs[(uts-1) * EXTENSION_FACTOR * user_num]))) % MODULUS for i in range(precision)]
    tc_evaluations.append([cn2 * zi2 * zd2 % MODULUS for cn2, zi2, zd2 in zip(c2_num_evaluations, z2_num_inv, z2_den_evaluations)])

    # constraint 3:t(i + uts*EXTENSION_FACTOR) = t(i + (uts-1)*EXTENSION_FACTOR) + t(i), i mod uts*EXTENSION_FACTOR == (uts-1)*EXTENSION_FACTOR， and i != last_step_position，
    # z3(x) = (x^user_num - G2^((uts-1) * EXTENSION_FACTOR * user_num))/(x - last_step_position)
    c3_num_evaluations = [(t_evaluations[(i + uts*EXTENSION_FACTOR) % precision] - t_evaluations[(i + (uts-1)*EXTENSION_FACTOR) % precision] - t_evaluations[i]) % MODULUS for i in range(precision)]
    z3_evaluations = [(xs[(i * user_num) % precision] - xs[(uts-1) * EXTENSION_FACTOR * user_num]) % MODULUS for i in range(precision)]
    z3_num_inv = f.multi_inv(z3_evaluations)
    z3_den_evaluations = [(xs[i] - last_step_position) % MODULUS for i in range(precision)]
    tc_evaluations.append([cn3 * zi3 * zd3 % MODULUS for cn3, zi3, zd3 in zip(c3_num_evaluations, z3_num_inv, z3_den_evaluations)])

    # constraint 4: t((uts-1)*EXTENSION_FACTOR) = 0， t(last_step_position) = sum_amounts[-1]
    # z4(x) = (x-xs[(uts-1)*EXTENSION_FACTOR])(x-last_step_position)
    interpolant = f.lagrange_interp_2([xs[(uts-1)*EXTENSION_FACTOR], last_step_position], [0, sum_amounts[-1]])
    i_evaluations = [f.eval_poly_at(interpolant, x) for x in xs]
    z4_poly = f.mul_polys([-xs[(uts-1)*EXTENSION_FACTOR], 1], [-last_step_position, 1])
    inv_z4_evaluations = f.multi_inv([f.eval_poly_at(z4_poly, x) for x in xs])    
    tc_evaluations.append([((t - i) * zi) % MODULUS for t, i, zi in zip(t_evaluations, i_evaluations, inv_z4_evaluations)])

    # coins constraints evaluations
    cc_evaluations = []
    for i in range(len(b_evaluations)):
        # cc_constraint 1: b_evaluations[i][j + uts*EXTENSION_FACTOR] = b_evaluations[i][j + (uts-1)*EXTENSION_FACTOR] + b_evaluations[i][j], j mod uts*EXTENSION_FACTOR == (uts-1)*EXTENSION_FACTOR，and j != last_step_position，
        # z(x) = (x^user_num - G2^((uts-1) * EXTENSION_FACTOR * user_num))/(x - last_step_position)
        cc1_num_evaluations = [(b_evaluations[i][(j + uts*EXTENSION_FACTOR) % precision] - b_evaluations[i][(j + (uts-1)*EXTENSION_FACTOR) % precision] - b_evaluations[i][j]) % MODULUS for j in range(precision)]
        zc1_evaluations = [(xs[(i * user_num) % precision] - xs[(uts-1) * EXTENSION_FACTOR * user_num]) % MODULUS for i in range(precision)]
        zc1_num_inv = f.multi_inv(zc1_evaluations)
        zc1_den_evaluations = [(xs[i] - last_step_position) % MODULUS for i in range(precision)]
        cc_evaluations.append([cn1 * zi1 * zd1 % MODULUS for cn1, zi1, zd1 in zip(cc1_num_evaluations, zc1_num_inv, zc1_den_evaluations)])

        # cc_constraints 2: b_evaluations[i](uts-1)*EXTENSION_FACTOR) = 0, b_evaluations[i](last_step_position) = sum_amount[i]
        # z(x) = (x-xs[(uts-1)*EXTENSION_FACTOR])(x-last_step_position)
        interpolant = f.lagrange_interp_2([xs[(uts-1)*EXTENSION_FACTOR], last_step_position], [0, sum_amounts[i]])
        i_evaluations = [f.eval_poly_at(interpolant, x) for x in xs]
        cc_evaluations.append([((t - i) * zi) % MODULUS for t, i, zi in zip(b_evaluations[i], i_evaluations, inv_z4_evaluations)])

    del i_evaluations, interpolant, zc1_den_evaluations, zc1_num_inv, zc1_evaluations, cc1_num_evaluations, inv_z4_evaluations,z4_poly, z3_den_evaluations, z3_num_inv, \
        z3_evaluations, c3_num_evaluations, z2_den_evaluations, z2_num_inv, z2_num_evaluations, c2_num_evaluations, z1_inv, z1_evaluations

    mtree_entries_data = get_entries([t_evaluations, b_evaluations, id_evaluations, tc_evaluations, cc_evaluations])
    user_random = [int.from_bytes(blake(r),'big') % MODULUS for r in get_entries([tc_evaluations, cc_evaluations])]
    m_tree_leaves = [blake(entry) for entry in get_entries([t_evaluations, b_evaluations, id_evaluations, user_random])]
    mtree = merkelize(m_tree_leaves)

    commit_time = time.time()
    print("commit in %d sec: " % (commit_time - start_time))

    # linearly combination
    G2_to_the_steps = f.exp(G2, 2 * steps)      
    powers = [1]                            
    for i in range(1, precision):
        powers.append(powers[-1] * G2_to_the_steps % MODULUS)
    
    l_evaluations = calculate_l(mtree[1], powers, [t_evaluations, b_evaluations, id_evaluations, tc_evaluations[0], tc_evaluations[2:], cc_evaluations], MODULUS)
    
    l_evaluations = [(l + c) % MODULUS for l ,c  in zip(l_evaluations, tc_evaluations[1])]

    l_mtree = merkelize(l_evaluations)
    samples = SPOT_CHECK_SECURITY_FACTOR
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples,
                                         exclude_multiples_of=EXTENSION_FACTOR)
    augmented_positions = sum([[x, (x + skips) % precision, (x + (uts-1)*skips) % precision, (x + uts*skips) % precision] for x in positions], [])
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots, the spot check Merkle proofs,
    # and low-degree proofs
    sum_proof = [steps,
        uts,
        mtree[1],
        l_mtree[1],
        mk_multi_branch(mtree, augmented_positions),
        [mtree_entries_data[i] for i in augmented_positions],
        mk_multi_branch(l_mtree, positions),
        prove_low_degree(l_evaluations, G2, 3*steps, MODULUS, exclude_multiples_of=EXTENSION_FACTOR)]

    save_data(data_path, sum_proof, mtree, mtree_entries_data, sum_amounts)

    return

# Verifies a STARK
def verify_por_proof(sum_amounts, proof):
    start_time = time.time()
    check_sum_amounts(sum_amounts, MODULUS)
    coins_num = len(sum_amounts) - 1
    steps, uts, m_root, l_root, main_branches, mtree_entries_data, linear_comb_branches, fri_proof = proof
    assert steps <= 2**32 // EXTENSION_FACTOR, "invalid steps: too large"
    assert is_a_power_of_2(steps), "invalid steps: should be a power of 2"

    precision = steps * EXTENSION_FACTOR
    user_num = steps // uts

    G2 = f.exp(NONRESIDUE, (MODULUS-1)//precision)
    skips = precision // steps
    G1 = f.exp(G2, skips)

    assert verify_low_degree_proof(l_root, G2, fri_proof, 3*steps, MODULUS, exclude_multiples_of=EXTENSION_FACTOR)

    verify_ldp_time = time.time()
    print("verify low degree in %.4f sec" % (verify_ldp_time - start_time))
    
    # Performs the spot checks
    k = [int.from_bytes(blake(m_root +i.to_bytes(32,'big')), 'big') % MODULUS for i in range(6 * coins_num + 10)]

    samples = SPOT_CHECK_SECURITY_FACTOR
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=EXTENSION_FACTOR)
    augmented_positions = sum([[x, (x + skips) % precision, (x + (uts-1)*skips) % precision, (x + uts*skips) % precision] for x in positions], [])
    last_step_position = f.exp(G2, (steps - 1) * skips)
 

    main_branch_leaves = verify_multi_branch(m_root, augmented_positions, main_branches)
    check_entry_hash(main_branch_leaves, mtree_entries_data, coins_num, MODULUS)

    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)

    for i, pos in enumerate(positions):
        x = f.exp(G2, pos)
        x_to_the_steps = f.exp(x, 2*steps)
        mbranch1 = mtree_entries_data[i*4]
        mbranch2 = mtree_entries_data[i*4+1]
        mbranch3 = mtree_entries_data[i*4+2]
        mbranch4 = mtree_entries_data[i*4+3]

        l_of_x = int.from_bytes(linear_comb_branch_leaves[i], 'big')

        t_of_x = int.from_bytes(mbranch1[:32], 'big')
        b_of_x = [int.from_bytes(mbranch1[32+32*i:64+32*i], 'big') for i in range(coins_num)]
        id_of_x = int.from_bytes(mbranch1[32+32*coins_num:64+32*coins_num], 'big')
        tc_of_x = [int.from_bytes(mbranch1[64+32*(coins_num+i):96+32*(coins_num+i)], 'big') for i in range(4)]
        cc_of_x = [int.from_bytes(mbranch1[192+32*(coins_num+i):224+32*(coins_num+i)], 'big') for i in range(2*coins_num)]

        t_of_skips_x = int.from_bytes(mbranch2[:32], 'big')
        t_of_uts_sub_1_skips_x = int.from_bytes(mbranch3[:32], 'big')
        t_of_uts_skips_x = int.from_bytes(mbranch4[:32], 'big')

        b_of_uts_sub_1_skips_x = [int.from_bytes(mbranch3[32+32*i:64+32*i], 'big') for i in range(coins_num)]
        b_of_uts_skips_x = [int.from_bytes(mbranch4[32+32*i:64+32*i], 'big') for i in range(coins_num)]

        # check constraint 1: t[uts*EXTENSION_FACTOR*i] = 0, 0<=i<=user_num-1, 
        # z1(x) = (x-xs[uts*EXTENSION_FACTOR*0])(x-xs[uts*EXTENSION_FACTOR*1])...(x-xs[uts*EXTENSION_FACTOR*(user_num-1)]) 
        # z1(x) = x^user_num - 1
        # t[i] = c1[i] * z1[i]
        z1 = f.exp(x, user_num) - 1
        assert((t_of_x - tc_of_x[0] * z1) % MODULUS == 0)

        # check constraint 2:  (t[i + EXTENSION_FACTOR] - 3*t[i])*(t[i + EXTENSION_FACTOR] - 3*t[i] - 1)*(t[i + EXTENSION_FACTOR] - 3*t[i] - 2) = 0, 
        # 0<=i<=steps-1 && (i mod uts*EXTENSION_FACTOR != {(uts-2)*EXTENSION_FACTOR,(uts-1)*EXTENSION_FACTOR}, i in range(precision)
        # z3(x) = (x^steps -1)/((x^user_num -  G2^((uts-2)*EXTENSION_FACTOR*user_num))(x^user_num - G2^((uts-1)*EXTENSION_FACTOR*user_num)))
        # (t[i + EXTENSION_FACTOR] - 3*t[i])*(t[i + EXTENSION_FACTOR] - 3*t[i] - 1)*(t[i + EXTENSION_FACTOR] - 3*t[i] - 2) = c3[i] * z3[i]
        z2 = f.div(f.exp(x, steps) - 1, 
                (f.exp(x, user_num) - f.exp(G2, (uts-2) * EXTENSION_FACTOR * user_num)) * (f.exp(x, user_num) - f.exp(G2, (uts-1) * EXTENSION_FACTOR * user_num)))
        assert(((t_of_skips_x - 3 * t_of_x) * (t_of_skips_x - 3 * t_of_x - 1) * (t_of_skips_x - 3 * t_of_x - 2) - tc_of_x[1] * z2) % MODULUS == 0)

        # check constraint 3: t(i + uts*EXTENSION_FACTOR) = t(i + (uts-1)*EXTENSION_FACTOR) + t(i),i mod uts*EXTENSION_FACTOR == (uts-1)*EXTENSION_FACTOR，i != last_step_position
        # z4(x) = (x^user_num - G2^((uts-1) * EXTENSION_FACTOR * user_num))/(x - last_step_position)
        # t(i + uts*EXTENSION_FACTOR) - t(i + (uts-1)*EXTENSION_FACTOR) - t(i) = c4[i] * z4
        z3 = f.div(f.exp(x, user_num) - f.exp(G2, (uts-1) * EXTENSION_FACTOR * user_num), x - last_step_position)
        assert((t_of_uts_skips_x - t_of_uts_sub_1_skips_x - t_of_x - tc_of_x[2] * z3) % MODULUS == 0)

        # check constraint 4: t((uts-1)*EXTENSION_FACTOR) = 0， t(last_step_position) = sum_amounts[-1]
        # z5(x) = (x-xs[(uts-1)*EXTENSION_FACTOR])(x-last_step_position)
        # t[i] - interpolant_value = c5[i] * z5[i]
        interpolant = f.lagrange_interp_2([f.exp(G2, (uts-1)*EXTENSION_FACTOR), last_step_position], [0, sum_amounts[-1]])
        interpolant_value = f.eval_poly_at(interpolant, x)
        z4_poly = f.mul_polys([-f.exp(G2, ((uts-1)*EXTENSION_FACTOR)), 1], [-last_step_position, 1])
        z4 = f.eval_poly_at(z4_poly, x)
        assert((t_of_x - interpolant_value - tc_of_x[3] * z4) % MODULUS == 0)

        # check coins constraint:
        for i in range(coins_num):
            # check cc_constraint 1: b_evaluations[i][j + uts*EXTENSION_FACTOR] = b_evaluations[i][j + (uts-1)*EXTENSION_FACTOR] + b_evaluations[i][j]
            # 当 j mod uts*EXTENSION_FACTOR == (uts-1)*EXTENSION_FACTOR，and j != last_step_position
            assert((b_of_uts_skips_x[i] - b_of_uts_sub_1_skips_x[i] - b_of_x[i] - cc_of_x[2*i] * z3) % MODULUS == 0)

            # cc_constraints 2: b_evaluations[i](uts-1)*EXTENSION_FACTOR) = 0, b_evaluations[i](last_step_position) = 0
            interpolant = f.lagrange_interp_2([f.exp(G2, (uts-1)*EXTENSION_FACTOR), last_step_position], [0, sum_amounts[i]])
            interpolant_value = f.eval_poly_at(interpolant, x)
            assert((b_of_x[i] - interpolant_value - cc_of_x[2*i+1] * z4) % MODULUS == 0)

        # Check correctness of the linear combination
        # l = calculate_l(kzg_commitment[0].n, [x_to_the_steps], [[[t_of_x]], [[b_of_x]],[id_of_x],[tc_of_x[0]],[tc_of_x[2:]],[cc_of_x]])
        assert verify_l(k, x_to_the_steps, l_of_x, [t_of_x, b_of_x, id_of_x, tc_of_x[0], tc_of_x[2:], cc_of_x], tc_of_x[1], MODULUS)

    print('Verified %d consistency checks' % SPOT_CHECK_SECURITY_FACTOR)
    print('Verified sum proof in %.4f sec' % (time.time() - start_time))
    return True
