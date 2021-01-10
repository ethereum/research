from poly_gcd import PrimeFieldExtended
from fft import fft, shift_poly, expand_root_of_unity
from random import randint
from time import time
import py_ecc.optimized_bls12_381 as b

MODULUS = b.curve_order

PRIMITIVE_ROOT_OF_UNITY = 5

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

def make_leaf(domain: list, indices: list) -> list:
    # legacy approach, requires many more tiny array allocations
    # ps = []
    # for i, x in enumerate(zero_vector):
    #     if x == 0:
    #         term = [-domain[i], 1]
    #         if len(ps) > 0 and primefield.degree(ps[-1]) < 63:  # extending the leaf
    #             ps[-1] = primefield.mul_polys(ps[-1], term)
    #         else:
    #             ps.append(term)  # new leaf
    out = [0]*(len(indices)) + [1]
    for i, v in enumerate(indices):
        # -domain[v] shortened to D
        # current poly:  a,     b,    c,    d,    e,   ...,    k,   1
        # mul term:      D,     1
        # result:        Da,   Db,   Dc,   Dd,   De,   ...,   Dk,   D,
        #            +   0,     a,    b,    c,    d,   ...,    l,   k,   1
        # So what we do:
        #  0. starting array must be prepared with trailing 1 that is never touched.
        #  1. append D to current poly (last 1 we are lagging, multiplied by D). Add value of i-1 if it exists.
        #  2. iterate through current poly backwards from near (excl) end, until (excl) first value
        #    2.1. multiply value with D
        #    2.2  add value of i-1
        #  3. first value only has work like 2.1: multiply with D
        out[i] = -domain[v]
        if i > 0:
            out[i] += out[i-1]
            out[i] %= primefield.modulus
            for j in range(i-1, 0, -1):
                out[j] = (out[j] * -domain[v]) + out[j-1]
                out[j] %= primefield.modulus
            out[0] *= -domain[v]
            out[0] %= primefield.modulus
    return out


def zero_polynomial_via_multiplication(root_of_unity, zero_vector):
    reduction_factor = 4
    time_a = time()
    domain = expand_root_of_unity(root_of_unity, primefield.modulus)
    time_b = time()
    print("time spent on zero poly domain calc: %.4f" % (time_b - time_a))
    indices = [i for i, x in enumerate(zero_vector) if x == 0]
    # the resulting leaf is 1 larger, so we want to stop at 31 indices per leaf
    batch_width = 31
    leaf_batches = [indices[i:min(len(indices), i+batch_width)] for i in range(0, len(indices), batch_width)]
    ps = [make_leaf(domain, leaf_batch) for leaf_batch in leaf_batches]
    time_c = time()
    print("time spent on leaf preparation %.4f" % (time_c - time_b))
    if len(ps) == 0:
        raise Exception("no zeroes in zero vector")
    while len(ps) > 1:
        # split into two (almost) equal length polys when the length gets small
        if reduction_factor < len(ps) < 2 * reduction_factor:
            psnew = []
            psnew.append(primefield.mul_many_polys_with_precomputed_domain(ps[:len(ps) // 2], domain))
            psnew.append(primefield.mul_many_polys_with_precomputed_domain(ps[len(ps) // 2:], domain))
        else:
            psnew = []
            for i in range(0, len(ps), reduction_factor):
                l = ps[i:min(len(ps), i+reduction_factor)]
                psnew.append(primefield.mul_many_polys_with_precomputed_domain(l, domain))
        ps = psnew

    time_d = time()
    print("time spent on leaf reduction %.4f" % (time_d - time_c))

    zero_poly = ps[0]
    assert primefield.degree(zero_poly) == len(list(filter(lambda x: x == 0, zero_vector)))
    r = fft(zero_poly, primefield.modulus, root_of_unity)

    time_e = time()
    print("time spent on fft of zero poly %.4f" % (time_e - time_d))

    assert all(a == 0 and b == 0 or a != 0 and b != 0 for a, b in zip(zero_vector, r))
    return r, zero_poly

def reconstruct_polynomial_from_samples(root_of_unity, samples, zero_polynomial_function):
    zero_vector = [0 if x is None else 1 for x in samples]

    time_a = time()
    zero_eval, zero_poly = zero_polynomial_function(root_of_unity, zero_vector)
    time_b = time()

    poly_evaluations_with_zero = [(0 if x is None else x) * y for x, y in zip(samples, zero_eval)]
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
    print("Reconstructed data using zero_polynomial_via_multiplication in {0:.5f} s (of which constructing zero poly: {1:.5f} s)".format(time_b - time_a, zero_time_a))
    assert reconstructed_data == data

    # reconstructed_data2, zero_time_b = reconstruct_polynomial_from_samples(ROOT_OF_UNITY, samples, zero_polynomial_via_gcd)
    # time_c = time()
    # print("Reconstructed data using zero_polynomial_via_gcd in {0:.2f} s (of which constructing zero poly: {1:.2f} s)".format(time_c - time_b, zero_time_b))
    # assert reconstructed_data == data
