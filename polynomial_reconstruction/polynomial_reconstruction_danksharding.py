from poly_gcd import PrimeFieldExtended
from fft import fft, shift_poly
import random
from time import time
import py_ecc.optimized_bls12_381 as b
from polynomial_reconstruction import reconstruct_polynomial_from_samples, zero_polynomial_via_multiplication

MODULUS = b.curve_order

PRIMITIVE_ROOT_OF_UNITY = 7

primefield = PrimeFieldExtended(MODULUS, PRIMITIVE_ROOT_OF_UNITY)

assert pow(PRIMITIVE_ROOT_OF_UNITY, MODULUS - 1, MODULUS) == 1
assert pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1) // 2, MODULUS) != 1

N = 13 # 2^13 data
M = 4 # 2^4 sample size
n = 2**N # 8192 for danksharding
m = 2**M # 16 sample size
ROOT_OF_UNITY = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1)//n, MODULUS)
ROOT_OF_UNITY_S = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS-1)//(2**(N - M)), MODULUS)
ROOT_OF_UNITY_SS = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS-1)//(2**M), MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(n)]

def is_power_of_two(x):
    return x > 0 and x & (x-1) == 0

def reverse_bit_order(n, order):
    """
    Reverse the bit order of an integer n
    """
    assert is_power_of_two(order)
    # Convert n to binary with the same number of bits as "order" - 1, then reverse its bit order
    return int(('{:0' + str(order.bit_length() - 1) + 'b}').format(n)[::-1], 2)


def list_to_reverse_bit_order(l):
    """
    Convert a list between normal and reverse bit order. This operation is idempotent.
    """
    return [l[reverse_bit_order(i, len(l))] for i in range(len(l))]

rbo = list_to_reverse_bit_order([i for i in range(n)])
rbo_s = list_to_reverse_bit_order([i for i in range(2 ** (N - M))])
rbo_ss = list_to_reverse_bit_order([i for i in range(m)])

if __name__ == "__main__":
    data = fft([random.randrange(MODULUS) for i in range(2**(N-1))],
                    MODULUS, ROOT_OF_UNITY)

    # discard half data
    indices = [i for i in range(2 ** (N - M))] # in samples
    random.shuffle(indices)
    avail_indices=indices[:len(indices)//2]
    missing_indices=indices[len(indices)//2:]
    
    erased_data = data[:]
    for i in missing_indices:
        for j in range(m):
            erased_data[rbo[i*m + j]] = None

    ############################################
    # Direct recovery using Vitalik's algorithm
    ############################################
    a = time()
    outdata_a, zero_time_a = reconstruct_polynomial_from_samples(ROOT_OF_UNITY, erased_data, zero_polynomial_via_multiplication)
    assert outdata_a == data
    print("Recovery in dataset of size %i done in time %.4f (of which constructing zero poly: %.2f s)" %
            (n, time() - a, zero_time_a))

    ####################################################
    # Recover zpoly in smaller group of order n/m with shifting parameter based on Danksharding encoding
    ####################################################
    a = time()
    # calcuate the zpoly in the smaller group
    zvec = [1] * 2 ** (N - M)
    for i in missing_indices:
        zvec[rbo_s[i]] = 0 # missing
    _, zs = zero_polynomial_via_multiplication(ROOT_OF_UNITY_S, zvec)

    # extend to the full group of order n
    z = []
    for i in zs:
        z.append(i)
        z.extend([0] * (m - 1))
    z_eval = fft(z, MODULUS, ROOT_OF_UNITY)
    zero_time_b = time() - a

    outdata_b, _ = reconstruct_polynomial_from_samples(ROOT_OF_UNITY, erased_data, lambda x, y: (z_eval, z))
    assert outdata_b == data
    print("Recovery in dataset of size %i with optimized zpoly done in time %.4f (of which constructing zero poly: %.2f s)" %
            (2**N, time() - a, zero_time_b))

    ####################################################
    # Optimized recovery on danksharding encoding
    ####################################################
    # pre-compute root of unit list
    a = time()
    
    # prepare shared variables
    # calcuate the zpoly in the smaller group
    zvec = [1] * 2 ** (N - M)
    for i in missing_indices:
        zvec[rbo_s[i]] = 0 # missing
    zs_evals, zs = zero_polynomial_via_multiplication(ROOT_OF_UNITY_S, zvec)
    shifted_zs = shift_poly(zs, MODULUS, PRIMITIVE_ROOT_OF_UNITY)
    eval_shifted_zs = fft(shifted_zs, MODULUS, ROOT_OF_UNITY_S)
    inv_eval_shifted_zs = primefield.multi_inv(eval_shifted_zs)
    prep_time = time() - a

    # IFFT all available samples
    coeffs = []
    for i in avail_indices:
        coeff = fft([erased_data[rbo[i * m + j]] for j in rbo_ss], MODULUS, ROOT_OF_UNITY_SS, True)
        coeff = [c * DOMAIN[-rbo_s[i] * k] % MODULUS for k, c in enumerate(coeff)]
        coeffs.extend(coeff)

    # recover pre-FFT values of the missing samples
    recs = []
    for i in range(m):
        ys = [None] * len(indices)
        for j, y in zip(avail_indices, coeffs[i::m]):
            ys[rbo_s[j]] = y

        outdata_s, _ = reconstruct_polynomial_from_samples(ROOT_OF_UNITY_S, ys, lambda x, y: (zs_evals, zs),
                                                            shifted_zero_poly=shifted_zs,
                                                            eval_shifted_zero_poly=eval_shifted_zs,
                                                            inv_eval_shifted_zero_poly=inv_eval_shifted_zs)
        recs.extend(outdata_s)

    # FFT to get the missing samples
    outdata_c = erased_data[:]
    for i in missing_indices:
        coeff = recs[rbo_s[i]::len(indices)]
        coeff = [c * DOMAIN[rbo_s[i] * k] % MODULUS for k, c in enumerate(coeff)]
        outdata_s = fft(coeff, MODULUS, ROOT_OF_UNITY_SS)
        for j, d in enumerate(outdata_s):
            outdata_c[rbo[i * m + rbo_ss[j]]] = d
    assert outdata_c == data
    print("Recovery in dataset of size %i with optimized recovery done in time %.4f (of which prepare time %.2f)" %
            (2**N, time() - a, prep_time))
