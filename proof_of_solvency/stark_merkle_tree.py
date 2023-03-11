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
from mimc_stark.utils import get_power_cycle, get_pseudorandom_indices, is_a_power_of_2



modulus = 2**256 - 2**32 * 351 + 1
f = PrimeField(modulus)
nonresidue = 7

spot_check_security_factor = 40
extension_factor = 4
# user trace size, user's net balance will be less than 2**(ust-2)
uts = 16

def get_sum_trace(balances):
    user_num = len(balances)//uts

    trace = [0]*len(balances)
    sum = 0
    for i in range(len(balances)):
        sum += balances[i]
    average = sum//user_num
    trace[uts-2] = balances[uts-2]
    trace[uts-1] = (trace[uts-2] - average) % modulus
    for i in range(1,user_num):
        trace[uts*i+(uts-2)] = balances[uts*i+(uts-2)]
        trace[uts*i+uts-1] = (trace[uts*(i-1)+uts-1] + trace[uts*i+(uts-2)] - average) % modulus
    for i in range(user_num):
        for j in range(uts-3):
            trace[uts*i+(uts-3)-j] = trace[uts*i+(uts-2)-j] // 2
    return trace, average, sum

def mk_por_proof(id, balances):
    start_time = time.time()
    # steps = uts * USER_NUM
    steps = len(balances)
    user_num = steps // uts
    precision = steps * extension_factor
    print("precision", precision)
    print("steps", steps)
    print("user number", user_num)
    

    # G2^precision mod modulus = 1
    G2 = f.exp(nonresidue, (modulus-1)//precision)
    skips = precision // steps

    # G1^steps mod modulus = 1
    G1 = f.exp(G2, skips)

    # Powers of the higher-order root of unity generate xs[1, G2, G2^2, ... G2^(precision-1)]
    xs = get_power_cycle(G2, modulus)
    # The corresponding value of xs in the extension field for the final step
    last_step_position = xs[(steps-1)*extension_factor]

    sum_trace, average, _ = get_sum_trace(balances)

    # Interpolate the computational trace into a polynomial P, with each step
    # along a successive power of G1
    # Compute the interpolation polynomial of degree steps - 1 using IFFT
    # based on the computation trace and the generator point G1.
    t_polynomial = fft(sum_trace, modulus, G1, inv=True)

    # Compute the value in the extension field using FFT based on the polynomial, 
    # extension field, and G2
    t_evaluations = fft(t_polynomial, modulus, G2)

    b_polynomial = fft(balances, modulus, G1, inv=True)
    b_evaluations = fft(b_polynomial, modulus, G2)

    id_evaluations = sum([[x]+[0]*(extension_factor-1) for x in id], [])


    # constraint 1: t[uts*extension_factor*i] = 0, 0<=i<=user_num-1, 
    # z1(x) = (x-xs[uts*extension_factor*0])(x-xs[uts*extension_factor*1])...(x-xs[uts*extension_factor*(user_num-1)]) 
    # z1(x) = x^user_num - 1
    z1_evaluations = [xs[(i * user_num) % precision] - 1 for i in range(precision)]
    z1_inv = f.multi_inv(z1_evaluations)
    c1_evaluations = [tp1 * zi1 % modulus for tp1, zi1 in zip(t_evaluations, z1_inv)]
    # print("c1_poly_degree", f.get_poly_degree_exclude_multiples_of(c1_evaluations, G2, [uts*extension_factor,[0]]))

    # constraint 2: t[uts*extension_factor*i+(uts-2)*extension_factor] = b(uts*extension_factor*i+(uts-2)*extension_factor), 0<=i<=user_num-1
    # z2(x) = (x-xs[uts*extension_factor*0+(uts-2)*extension_factor])(x-xs[uts*extension_factor*1+(uts-2)*extension_factor])...(x-xs[uts*extension_factor*(user_num-1)+(uts-2)*extension_factor])
    # z2(x) = x^user_num - G2^((uts-2)*extension_factor*user_num)
    c2_num_evaluations = [(t_evaluations[i] - b_evaluations[i]) % modulus for i in range(precision)]
    z2_evaluations = [(xs[(i * user_num) % precision] - xs[(uts-2) * extension_factor * user_num]) % modulus for i in range(precision)]
    z2_inv = f.multi_inv(z2_evaluations)
    c2_evaluations = [cn2 * zi2 % modulus for cn2, zi2 in zip(c2_num_evaluations, z2_inv)]

    # constraint 3:  (t[i + extension_factor] - 2*t[i])*(t[i + extension_factor] - 2*t[i] - 1) = 0, 
    # 0<=i<=steps-1 && (i mod uts*extension_factor != {(uts-2)*extension_factor,(uts-1)*extension_factor}, i in range(precision)
    # z3(x) = (x^steps -1)/((x^user_num -  G2^((uts-2)*extension_factor*user_num))(x^user_num - G2^((uts-1)*extension_factor*user_num)))
    c3_num_evaluations = [((t_evaluations[(i + extension_factor) % precision] - 2 * t_evaluations[i]) * 
                      (t_evaluations[(i + extension_factor) % precision] - 2 * t_evaluations[i] - 1)) % modulus for i in range(precision)]
    
    z3_num_evaluations = [(xs[(i * steps) % precision] - 1) % modulus for i in range(precision)]
    z3_num_inv = f.multi_inv(z3_num_evaluations)
    z3_den_evaluations = [(((xs[(i * user_num) % precision] - xs[(uts-2) * extension_factor * user_num])) * 
                           ((xs[(i * user_num) % precision] - xs[(uts-1) * extension_factor * user_num]))) % modulus for i in range(precision)]
    c3_evaluations = [cn3 * zi3 * zd3 % modulus for cn3, zi3, zd3 in zip(c3_num_evaluations, z3_num_inv, z3_den_evaluations)]

    # constraint 4:t(i + uts*extension_factor) = t(i + (uts-1)*extension_factor) + t(i) - average, when i mod uts*extension_factor == (uts-1)*extension_factor and i != last_step_position is satisfied simultaneously，
    # z4(x) = (x^user_num - G2^((uts-1) * extension_factor * user_num))/(x - last_step_position)
    c4_num_evaluations = [(t_evaluations[(i + uts*extension_factor) % precision] - t_evaluations[(i + (uts-1)*extension_factor) % precision] - t_evaluations[i] + average) % modulus for i in range(precision)]
    z4_evaluations = [(xs[(i * user_num) % precision] - xs[(uts-1) * extension_factor * user_num]) % modulus for i in range(precision)]
    z4_num_inv = f.multi_inv(z4_evaluations)
    z4_den_evaluations = [(xs[i] - last_step_position) % modulus for i in range(precision)]
    c4_evaluations = [cn4 * zi4 * zd4 % modulus for cn4, zi4, zd4 in zip(c4_num_evaluations, z4_num_inv, z4_den_evaluations)]

    # constraint 5: t((uts-1)*extension_factor) = b((uts-2)*extension_factor) - average， t(last_step_position) = 0
    # z5(x) = (x-xs[(uts-1)*extension_factor])(x-last_step_position)
    interpolant = f.lagrange_interp_2([xs[(uts-1)*extension_factor], last_step_position], [(b_evaluations[(uts-2)*extension_factor] - average) % modulus, 0])
    i_evaluations = [f.eval_poly_at(interpolant, x) for x in xs]
    z5_poly = f.mul_polys([-xs[(uts-1)*extension_factor], 1], [-last_step_position, 1])

    inv_z5_evaluations = f.multi_inv([f.eval_poly_at(z5_poly, x) for x in xs])    
    c5_evaluations = [((t - i) * zi) % modulus for t, i, zi in zip(t_evaluations, i_evaluations, inv_z5_evaluations)]

    # Compute their Merkle root
    mtree = merkelize([tval.to_bytes(32, 'big') +   # Convert an integer value to a 32-byte length 'bytes' type using big-endian byte order
                       bval.to_bytes(32, 'big') +
                       idval.to_bytes(32, 'big') +
                       c1val.to_bytes(32, 'big') +
                       c2val.to_bytes(32, 'big') +
                       c3val.to_bytes(32, 'big') +
                       c4val.to_bytes(32, 'big') +
                       c5val.to_bytes(32, 'big') for
                       tval, bval, idval, c1val, c2val, c3val, c4val, c5val in 
                       zip(t_evaluations, b_evaluations, id_evaluations, c1_evaluations, c2_evaluations, c3_evaluations, c4_evaluations, c5_evaluations)])

    # linearly combine c_poly and bn_poly, so we can prove the low degree with one combined polynomial
    k = [int.from_bytes(blake(mtree[1] + bytes([i])), 'big') for i in range(14)]

    # G2_to_the_steps  = G2^steps
    G2_to_the_steps = f.exp(G2, steps)
    # powers[i] = G2_to_the_steps^i = (G2^i)^steps
    powers = [1]
    for i in range(1, precision):
        powers.append(powers[-1] * G2_to_the_steps % modulus)
    
    l_evaluations = [(k[0] * t_evaluations[i] + k[1] * t_evaluations[i] * powers[i] + 
                    k[2] * b_evaluations[i] + k[3] * b_evaluations[i] * powers[i] +     
                    k[4] * id_evaluations[i] + k[5] * id_evaluations[i] * powers[i] +     
                    k[6] * c1_evaluations[i] + k[7] * c1_evaluations[i] * powers[i] +     
                    k[8] * c2_evaluations[i] + k[9] * c2_evaluations[i] * powers[i] +     
                    k[10] * c4_evaluations[i] + k[11] * c4_evaluations[i] * powers[i] +     
                    k[12] * c5_evaluations[i] + k[13] * c5_evaluations[i] * powers[i] +     
                    c3_evaluations[i]) % modulus for i in range(precision)]

    l_mtree = merkelize(l_evaluations)

    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_mtree[1], precision, samples, exclude_multiples_of=extension_factor)
    # Extend the position                                  
    augmented_positions = sum([[x, (x + skips) % precision, (x + (uts-1)*skips) % precision, (x + uts*skips) % precision] for x in positions], [])
    print('Computed %d spot checks' % samples)

    # Return the Merkle roots of P and D, the spot check Merkle proofs,
    # and low-degree proofs of P and D
    proof = [mtree[1],
         l_mtree[1],
         mk_multi_branch(mtree, augmented_positions),
         mk_multi_branch(l_mtree, positions),
         prove_low_degree(l_evaluations, G2, 2*steps, modulus, exclude_multiples_of=extension_factor)]

    print("STARK computed in %.4f sec" % (time.time() - start_time))

    return proof, mtree

# Verifies a STARK
def verify_por_proof(steps, sum_amount, b0, proof):

    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    start_time = time.time()
    assert steps <= 2**28 // extension_factor
    assert is_a_power_of_2(steps)

    precision = steps * extension_factor
    user_num = steps // uts
    average = sum_amount // user_num

    # G2^precision mod modulus= 1
    G2 = f.exp(nonresidue, (modulus-1)//precision)
    skips = precision // steps

    # G1^steps mod modulus = 1
    G1 = f.exp(G2, skips)

    assert verify_low_degree_proof(l_root, G2, fri_proof, 2*steps, modulus, exclude_multiples_of=extension_factor)

    # Performs the spot checks
    k = [int.from_bytes(blake(m_root + bytes([i])), 'big') for i in range(14)]

    samples = spot_check_security_factor
    positions = get_pseudorandom_indices(l_root, precision, samples,
                                         exclude_multiples_of=extension_factor)
    augmented_positions = sum([[x, (x + skips) % precision, (x + (uts-1)*skips) % precision, (x + uts*skips) % precision] for x in positions], [])
    last_step_position = f.exp(G2, (steps - 1) * skips)
    # Return the information of the leaf node
    main_branch_leaves = verify_multi_branch(m_root, augmented_positions, main_branches)

    linear_comb_branch_leaves = verify_multi_branch(l_root, positions, linear_comb_branches)

    for i, pos in enumerate(positions):
        x = f.exp(G2, pos)
        x_to_the_steps = f.exp(x, steps)
        mbranch1 = main_branch_leaves[i*4]
        mbranch2 = main_branch_leaves[i*4+1]
        mbranch3 = main_branch_leaves[i*4+2]
        mbranch4 = main_branch_leaves[i*4+3]
        l_of_x = int.from_bytes(linear_comb_branch_leaves[i], 'big')

        t_of_x = int.from_bytes(mbranch1[:32], 'big')
        b_of_x = int.from_bytes(mbranch1[32:64], 'big')
        id_of_x = int.from_bytes(mbranch1[64:96], 'big')
        c1_of_x = int.from_bytes(mbranch1[96:128], 'big')
        c2_of_x = int.from_bytes(mbranch1[128:160], 'big')
        c3_of_x = int.from_bytes(mbranch1[160:192], 'big')
        c4_of_x = int.from_bytes(mbranch1[192:224], 'big')
        c5_of_x = int.from_bytes(mbranch1[224:256], 'big')
        t_of_skips_x = int.from_bytes(mbranch2[:32], 'big')
        t_of_uts_sub_1_skips_x = int.from_bytes(mbranch3[:32], 'big')
        t_of_uts_skips_x = int.from_bytes(mbranch4[:32], 'big')


        # check constraint 1: t[uts*extension_factor*i] = 0, 0<=i<=user_num-1, 
        # z1(x) = (x-xs[uts*extension_factor*0])(x-xs[uts*extension_factor*1])...(x-xs[uts*extension_factor*(user_num-1)]) 
        # z1(x) = x^user_num - 1
        # t[i] = c1[i] * z1[i]
        z1 = f.exp(x, user_num) - 1
        assert((t_of_x - c1_of_x * z1) % modulus == 0)

        # check constraint 2: t[uts*extension_factor*i+(uts-2)*extension_factor] = b(uts*extension_factor*i+(uts-2)*extension_factor), 0<=i<=user_num-1
        # z2(x) = (x-xs[uts*extension_factor*0+(uts-2)*extension_factor])(x-xs[uts*extension_factor*1+(uts-2)*extension_factor])...(x-xs[uts*extension_factor*(user_num-1)+(uts-2)*extension_factor])
        # z2(x) = x^user_num - G2^((uts-2)*extension_factor*user_num)
        # t[i] - b[i] = c2[i] * z2[i]
        z2 = f.exp(x, user_num) - f.exp(G2, (uts-2) * extension_factor * user_num)
        assert((t_of_x - b_of_x - c2_of_x * z2) % modulus == 0)


        # check constraint 3:  (t[i + extension_factor] - 2*t[i])*(t[i + extension_factor] - 2*t[i] - 1) = 0, 
        # 0<=i<=steps-1 && (i mod uts*extension_factor != {(uts-2)*extension_factor,(uts-1)*extension_factor}, i in range(precision)
        # z3(x) = (x^steps -1)/((x^user_num -  G2^((uts-2)*extension_factor*user_num))(x^user_num - G2^((uts-1)*extension_factor*user_num)))
        # (t[i + extension_factor] - 2*t[i])*(t[i + extension_factor] - 2*t[i] - 1) = c3[i] * z3[i]
        z3 = f.div(f.exp(x, steps) - 1, 
                (f.exp(x, user_num) - f.exp(G2, (uts-2) * extension_factor * user_num)) * (f.exp(x, user_num) - f.exp(G2, (uts-1) * extension_factor * user_num)))
        assert(((t_of_skips_x - 2 * t_of_x) * (t_of_skips_x - 2 * t_of_x - 1) - c3_of_x * z3) % modulus == 0)

        # check constraint 4: t(i + uts*extension_factor) = t(i + (uts-1)*extension_factor) + t(i) - average, i mod uts*extension_factor == (uts-1)*extension_factor, i != last_step_position，
        # z4(x) = (x^user_num - G2^((uts-1) * extension_factor * user_num))/(x - last_step_position)
        # t(i + uts*extension_factor) - t(i + (uts-1)*extension_factor) - t(i) + average = c4[i] * z4
        z4 = f.div(f.exp(x, user_num) - f.exp(G2, (uts-1) * extension_factor * user_num), x - last_step_position)
        assert((t_of_uts_skips_x - t_of_uts_sub_1_skips_x - t_of_x + average - c4_of_x * z4) % modulus == 0)

        # check constraint 5: t((uts-1)*extension_factor) = b((uts-2)*extension_factor) - average， t(last_step_position) = 0
        # z5(x) = (x-xs[(uts-1)*extension_factor])(x-last_step_position)
        # t[i] - interpolant_value = c5[i] * z5[i]
        interpolant = f.lagrange_interp_2([f.exp(G2, (uts-1)*extension_factor), last_step_position], [(b0 - average) % modulus, 0])
        interpolant_value = f.eval_poly_at(interpolant, x)
        z5_poly = f.mul_polys([-f.exp(G2, ((uts-1)*extension_factor)), 1], [-last_step_position, 1])
        z5 = f.eval_poly_at(z5_poly, x)
        assert((t_of_x - interpolant_value - c5_of_x * z5) % modulus == 0)

        # Check correctness of the linear combination
        assert ((l_of_x - k[0] * t_of_x - k[1] * t_of_x * x_to_the_steps -
                k[2] * b_of_x - k[3] * b_of_x * x_to_the_steps -
                k[4] * id_of_x - k[5] * id_of_x * x_to_the_steps -
                k[6] * c1_of_x - k[7] * c1_of_x * x_to_the_steps -
                k[8] * c2_of_x - k[9] * c2_of_x * x_to_the_steps -
                k[10] * c4_of_x - k[11] * c4_of_x * x_to_the_steps -
                k[12] * c5_of_x - k[13] * c5_of_x * x_to_the_steps -
                c3_of_x) % modulus == 0)

    print('Verified %d consistency checks' % spot_check_security_factor)
    print('Verified STARK in %.4f sec' % (time.time() - start_time))
    return True

def mk_inclusion_proof(mtree, index):
    return mk_branch(mtree, (uts * index + (uts-2)) * extension_factor)

def verify_inclusion_proof(mroot, index, proof):
    verify_branch(mroot, (uts * index + (uts-2)) * extension_factor, proof)
    return True
