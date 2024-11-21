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

n = 2**15
ROOT_OF_UNITY = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1)//n, MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(n)]

def zero_polynomial_via_gcd(root_of_unity, zero_vector):
    e1 = [0 if x == 0 or i % 2 == 0 else 1 for i, x in enumerate(zero_vector)]
    e2 = [0 if x == 0 or i % 2 == 1 else 1 for i, x in enumerate(zero_vector)]
    p1 = fft(e1, primefield.modulus, root_of_unity, inv=True)
    p2 = fft(e2, primefield.modulus, root_of_unity, inv=True)
    zero_poly = primefield.fast_extended_euclidean_algorithm(p1, p2)[0]
    assert primefield.degree(zero_poly) == len(list(filter(lambda x: x == 0, zero_vector)))
    r = fft(zero_poly, primefield.modulus, root_of_unity)
    assert all(a == 0 and b == 0 or a != 0 and b != 0 for a, b in zip(zero_vector, r))
    return r, zero_poly


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
        if len(ps) < 2 * reduction_factor and len(ps) > reduction_factor:
            psnew = [primefield.mul_many_polys(ps[:len(ps) // 2]), primefield.mul_many_polys(ps[len(ps) // 2:])]
        else:
            psnew = [primefield.mul_many_polys(l) for l in zip(*[ps[i::reduction_factor] for i in range(reduction_factor)])]
            if len(ps) % reduction_factor != 0:
                psnew.append(primefield.mul_many_polys(ps[- (len(ps) % reduction_factor):]))
        ps = psnew
    zero_poly = ps[0]
    assert primefield.degree(zero_poly) == len(list(filter(lambda x: x == 0, zero_vector)))
    r = fft(zero_poly, primefield.modulus, root_of_unity)
    assert all(a == 0 and b == 0 or a != 0 and b != 0 for a, b in zip(zero_vector, r))
    return r, zero_poly

def reconstruct_polynomial_from_samples(root_of_unity, samples, zero_polynomial_function,
            shifted_zero_poly=None, eval_shifted_zero_poly=None, inv_eval_shifted_zero_poly=None):
    zero_vector = [0 if x is None else 1 for x in samples]

    time_a = time()
    zero_eval, zero_poly = zero_polynomial_function(root_of_unity, zero_vector)
    time_b = time()

    poly_evaluations_with_zero = [(0 if x is None else x) * y for x, y in zip(samples, zero_eval)]
    poly_with_zero = fft(poly_evaluations_with_zero, MODULUS, root_of_unity, inv=True)

    shift_factor = PRIMITIVE_ROOT_OF_UNITY
    shift_inv = primefield.inv(PRIMITIVE_ROOT_OF_UNITY)
    
    shifted_poly_with_zero = shift_poly(poly_with_zero, MODULUS, PRIMITIVE_ROOT_OF_UNITY)
    shifted_zero_poly = shifted_zero_poly or shift_poly(zero_poly, MODULUS, PRIMITIVE_ROOT_OF_UNITY)
    
    eval_shifted_poly_with_zero = fft(shifted_poly_with_zero, MODULUS, root_of_unity)
    eval_shifted_zero_poly = eval_shifted_zero_poly or fft(shifted_zero_poly, MODULUS, root_of_unity)
    
    if inv_eval_shifted_zero_poly:
        eval_shifted_reconstructed_poly = [primefield.mul(a, b) for a, b in
                                        zip(eval_shifted_poly_with_zero, inv_eval_shifted_zero_poly)]
    else:
        eval_shifted_reconstructed_poly = [primefield.div(a, b) for a, b in
                                        zip(eval_shifted_poly_with_zero, eval_shifted_zero_poly)]
    
    shifted_reconstructed_poly = fft(eval_shifted_reconstructed_poly, MODULUS, root_of_unity, inv=True)
    
    reconstructed_poly = shift_poly(shifted_reconstructed_poly, MODULUS, shift_inv)

    reconstructed_data = fft(reconstructed_poly, MODULUS, root_of_unity)
    
    assert all(x is None or x == y for x, y in zip(samples, reconstructed_data))
    
    return reconstructed_data, time_b - time_a


if __name__ == "__main__":
    poly = [i % 10 for i in range(n // 2)]
    data = fft(poly, MODULUS, ROOT_OF_UNITY)
    samples = data[:]
    for i in range(n//2):
        j = randint(0, n-1)
        while samples[j] == None:
            j = randint(0, n-1)
        samples[j] = None

    print("Computed data samples")
    time_a = time()
    reconstructed_data, zero_time_a = reconstruct_polynomial_from_samples(ROOT_OF_UNITY, samples, zero_polynomial_via_multiplication)
    time_b = time()
    print("Reconstructed data using zero_polynomial_via_multiplication in {0:.2f} s (of which constructing zero poly: {1:.2f} s)".format(time_b - time_a, zero_time_a))
    assert reconstructed_data == data

    reconstructed_data2, zero_time_b = reconstruct_polynomial_from_samples(ROOT_OF_UNITY, samples, zero_polynomial_via_gcd)
    time_c = time()
    print("Reconstructed data using zero_polynomial_via_gcd in {0:.2f} s (of which constructing zero poly: {1:.2f} s)".format(time_c - time_b, zero_time_b))
    assert reconstructed_data == data
