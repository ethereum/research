from poly_gcd import PrimeFieldExtended
from fft import fft, shift_poly
from random import randint
from time import time
import py_ecc.optimized_bls12_381 as b

MODULUS = b.curve_order

PRIMITIVE_ROOT_OF_UNITY = 7

primefield = PrimeFieldExtended(MODULUS, PRIMITIVE_ROOT_OF_UNITY)

assert pow(PRIMITIVE_ROOT_OF_UNITY, MODULUS - 1, MODULUS) == 1
assert pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1) // 2, MODULUS) != 1

n = 2**14
ROOT_OF_UNITY = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1)//n, MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(n)]

PREPARED_ZERO_POLYNOMIAL_INTERVAL = 1024

def zero_polynomial_via_multiplication(root_of_unity, zero_vector):
    reduction_factor = 4
    ps = [[1]]
    for i, x in enumerate(zero_vector):
        if x == 0:
            if primefield.degree(ps[-1]) < 63:
                ps[-1] = primefield.mul_polys(ps[-1], [-pow(root_of_unity, i, primefield.modulus), 1])
            else:
                ps.append([-pow(root_of_unity, i, primefield.modulus), 1])
    while len(ps) > 1:
        if len(ps) <= 2 * reduction_factor and len(ps) > reduction_factor:
            psnew = [primefield.mul_many_polys(ps[:len(ps) // 2]), primefield.mul_many_polys(ps[len(ps) // 2:])]
        elif len(ps) <= reduction_factor:
            r = primefield.mul_many_polys(ps, result_in_evaluation_form=True, size=len(zero_vector))
            assert all(a == 0 and b == 0 or a != 0 and b != 0 for a, b in zip(zero_vector, r))
            return r            
        else:
            psnew = [primefield.mul_many_polys(l) for l in zip(*[ps[i::reduction_factor] for i in range(reduction_factor)])]
            if len(ps) % reduction_factor != 0:
                psnew.append(primefield.mul_many_polys(ps[- (len(ps) % reduction_factor):]))
        ps = psnew

def interpolate_polynomial_without_zeroes(root_of_unity, samples):
    precomputable_zero_sample_blocks = 1
    while all(x == None for x in samples[- precomputable_zero_sample_blocks * PREPARED_ZERO_POLYNOMIAL_INTERVAL:]):
        precomputable_zero_sample_blocks += 1
    precomputable_zero_sample_blocks -= 1

    time_a = time()
    precomputable_zero_vector = [1] * (len(samples) - precomputable_zero_sample_blocks * PREPARED_ZERO_POLYNOMIAL_INTERVAL) \
                              + [0] * (precomputable_zero_sample_blocks * PREPARED_ZERO_POLYNOMIAL_INTERVAL)

    precomputable_zero_eval = zero_polynomial_via_multiplication(root_of_unity, precomputable_zero_vector)

    time_b = time()
    
    remaining_zero_vector = [0 if x is None else 1 for x in samples[:len(samples) - precomputable_zero_sample_blocks * PREPARED_ZERO_POLYNOMIAL_INTERVAL]] \
                          + [1] * (precomputable_zero_sample_blocks * PREPARED_ZERO_POLYNOMIAL_INTERVAL)
    
    remaining_zero_eval = zero_polynomial_via_multiplication(root_of_unity, remaining_zero_vector)

    time_c = time()

    zero_eval = [x * y % MODULUS for x, y in zip(precomputable_zero_eval, remaining_zero_eval)]
    zero_poly = fft(zero_eval, MODULUS, ROOT_OF_UNITY, inv=True)
    
    poly_evaluations_with_zero = [(0 if x is None else x) * y % MODULUS for x, y in zip(samples, zero_eval)]
    poly_with_zero = fft(poly_evaluations_with_zero, MODULUS, root_of_unity, inv=True)

    shift_factor = PRIMITIVE_ROOT_OF_UNITY
    shift_inv = primefield.inv(PRIMITIVE_ROOT_OF_UNITY)
    
    shifted_poly_with_zero = shift_poly(poly_with_zero, MODULUS, PRIMITIVE_ROOT_OF_UNITY)
    shifted_zero_poly = shift_poly(zero_poly, MODULUS, PRIMITIVE_ROOT_OF_UNITY)
    
    eval_shifted_poly_with_zero = fft(shifted_poly_with_zero, MODULUS, ROOT_OF_UNITY)
    eval_shifted_zero_poly = fft(shifted_zero_poly, MODULUS, ROOT_OF_UNITY)
    
    eval_shifted_reconstructed_poly = [primefield.div(a, b) for a, b in 
                                       zip(eval_shifted_poly_with_zero, eval_shifted_zero_poly)]
    
    shifted_reconstructed_poly = fft(eval_shifted_reconstructed_poly, MODULUS, ROOT_OF_UNITY, inv=True)
    
    reconstructed_poly = shift_poly(shifted_reconstructed_poly, MODULUS, shift_inv)

    reconstructed_data = fft(reconstructed_poly, MODULUS, ROOT_OF_UNITY)
    
    assert all(x is None or x == y for x, y in zip(samples, reconstructed_data))

    time_d = time()
    
    return reconstructed_data, time_b - time_a, time_c - time_b, time_d - time_c


if __name__ == "__main__":
    length = 8193

    poly = [10**i % MODULUS for i in range(length)]
    data = fft(poly, MODULUS, ROOT_OF_UNITY)
    samples = data[:length] + [None] * (n - length)

    print("Computed data samples")
    time_a = time()
    full_data, time1, time2, time3 = interpolate_polynomial_without_zeroes(ROOT_OF_UNITY, samples)
    time_b = time()
    print("Time to compute rough zero polynomial (up to blocks of size {0}, can be precomputed) {1:.2f} s".format(PREPARED_ZERO_POLYNOMIAL_INTERVAL, time1))
    print("Time to compute filling zero polynomial {0:.2f} s".format(time2))
    print("Time to get extended data samples {0:.2f} s".format(time3))

    assert full_data == data