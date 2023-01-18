import blst
from fft import fft
import kzg_proofs
from kzg_proofs import (
    MODULUS,
    check_proof_multi,
    generate_setup,
    commit_to_poly,
    list_to_reverse_bit_order,
    get_root_of_unity,
    reverse_bit_order,
    is_power_of_two,
    eval_poly_at,
    get_extended_data,
    P1_INF
)
from fk20_single import (
    toeplitz_part1,
    toeplitz_part2,
    toeplitz_part3
)

# FK20 Method to compute all proofs
# Toeplitz multiplication via http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html
# Multi proof method

def fk20_multi(polynomial, l, setup):
    """
    For a polynomial of size n, let w be a n-th root of unity. Then this method will return
    k=n/l KZG proofs for the points
        proof[0]: w^(0*l + 0), w^(0*l + 1), ... w^(0*l + l - 1)
        proof[0]: w^(0*l + 0), w^(0*l + 1), ... w^(0*l + l - 1)
        ...
        proof[i]: w^(i*l + 0), w^(i*l + 1), ... w^(i*l + l - 1)
        ...
    """

    n = len(polynomial)
    k = n // l
    assert is_power_of_two(n)
    assert is_power_of_two(l)
    assert k >= 1
    
    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    xext_fft = []
    for i in range(l):
        x = setup[0][n - l - 1 - i::-l] + [P1_INF.dup()]
        xext_fft.append(toeplitz_part1(x))

    hext_fft = [P1_INF.dup()] * 2 * k
    for i in range(l):

        toeplitz_coefficients = polynomial[- i - 1::l] + [0] * (k + 1) + polynomial[2 * l - i - 1: - l - i:l]

        # Compute the vector h from the paper using a Toeplitz matric multiplication
        hext_fft = [v.dup().add(w) for v, w in zip(hext_fft, toeplitz_part2(toeplitz_coefficients, xext_fft[i]))]
    
    h = toeplitz_part3(hext_fft)

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(k))


def fk20_multi_data_availability_optimized(polynomial, l, setup):
    """
    FK20 multi-proof method, optimized for dava availability where the top half of polynomial
    coefficients == 0
    """

    assert is_power_of_two(len(polynomial))
    n = len(polynomial) // 2
    k = n // l
    assert is_power_of_two(n)
    assert is_power_of_two(l)
    assert k >= 1

    assert all(x == 0 for x in polynomial[n:])
    reduced_polynomial = polynomial[:n]

    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    xext_fft = []
    for i in range(l):
        x = setup[0][n - l - 1 - i::-l] + [P1_INF.dup()]
        xext_fft.append(toeplitz_part1(x))

    add_instrumentation()

    hext_fft = [P1_INF.dup()] * 2 * k
    for i in range(l):

        toeplitz_coefficients = reduced_polynomial[- i - 1::l] + [0] * (k + 1) \
             + reduced_polynomial[2 * l - i - 1: - l - i:l]

        # Compute the vector h from the paper using a Toeplitz matric multiplication
        hext_fft = [v.dup().add(w) for v, w in zip(hext_fft, toeplitz_part2(toeplitz_coefficients, xext_fft[i]))]

    # Final FFT done after summing all h vectors
    h = toeplitz_part3(hext_fft)

    h = h + [P1_INF.dup()] * k

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(2 * k))


def data_availabilty_using_fk20_multi(polynomial, l, setup):
    """
    Computes all the KZG proofs for data availability checks. This involves sampling on the double domain
    and reordering according to reverse bit order
    """
    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    extended_polynomial = polynomial + [0] * n

    all_proofs = fk20_multi_data_availability_optimized(extended_polynomial, l, setup)

    return list_to_reverse_bit_order(all_proofs)


def add_instrumentation():
    global multiplication_count
    
    multiplication_count = 0

    # Add counter to multiply function for statistics
    b_multiply_ = blst.P1.mult
    def multiply_and_count(*args):
        global multiplication_count
        multiplication_count += 1

        return b_multiply_(*args)

    blst.P1.mult = multiply_and_count


if __name__ == "__main__":
    #uncomment to report time for the functions
    #from timer import chrono
    #generate_setup = chrono(generate_setup)
    #data_availabilty_using_fk20_multi = chrono(data_availabilty_using_fk20_multi)
    #check_proof_multi = chrono(check_proof_multi)

    import random
    from tqdm import tqdm

    MAX_DEGREE_POLY = MODULUS-1
    N_POINTS = 512


    polynomial = [random.randint(1, MAX_DEGREE_POLY) for _ in range(512)] 
    n = N_POINTS

    setup = generate_setup(random.getrandbits(256), n)

    commitment = commit_to_poly(polynomial, setup)

    l = 16
    
    all_proofs = data_availabilty_using_fk20_multi(polynomial, l, setup)
    print("All KZG proofs computed for data availability (supersampled by factor 2)")
    print("Required {0} G1 multiplications".format(multiplication_count))
    print(n, l, multiplication_count)

    # Now check all positions
    extended_data = get_extended_data(polynomial)

    for pos in range(2 * n // l):
        root_of_unity = get_root_of_unity(n * 2)
        x = pow(root_of_unity, reverse_bit_order(pos, 2 * n // l), MODULUS)
        ys = extended_data[l * pos:l * (pos + 1)]

        subgroup_root_of_unity = get_root_of_unity(l)
        coset = [x * pow(subgroup_root_of_unity, i, MODULUS) for i in range(l)]
        ys2 = [eval_poly_at(polynomial, z) for z in coset]
        assert list_to_reverse_bit_order(ys) == ys2

        assert check_proof_multi(commitment, all_proofs[pos], x, list_to_reverse_bit_order(ys), setup)
        print("Data availability sample check {0} passed".format(pos))
    
