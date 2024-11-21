import blst
from fft import fft
import kzg_proofs
from kzg_proofs import (
    MODULUS,
    check_proof_single,
    generate_setup,
    commit_to_poly,
    list_to_reverse_bit_order,
    get_root_of_unity,
    reverse_bit_order,
    is_power_of_two,
    eval_poly_at,
    P1_INF 
)

# FK20 Method to compute all proofs
# Toeplitz multiplication via http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html
# Single proof method

# A Toeplitz matrix is of the form
#
# t_0     t_(-1) t_(-2) ... t_(1-n)
# t_1     t_0    t_(-1) ... t_(2-n)
# t_2     t_1               .
# .              .          .
# .                 .       .
# .                    .    t(-1)
# t_(n-1)   ...       t_1   t_0
#
# The vector [t_0, t_1, ..., t_(n-2), t_(n-1), 0, t_(1-n), t_(2-n), ..., t_(-2), t_(-1)] 
# completely determines the Toeplitz matrix and is called the "toeplitz_coefficients" below


# The composition toeplitz_part3(toeplitz_part2(toeplitz_coefficients, toeplitz_part1(x)))
# compute the matrix-vector multiplication T * x
#
# The algorithm here is written under the assumption x = G1 elements, T scalars
#
# For clarity, vectors in "Fourier space" are written with _fft. So for example, the vector
# xext is the extended x vector (padded with zero), and xext_fft is its Fourier transform.


# Toeplitz matrix multiplication part 1 -- precompute (independent of polynomial)
def toeplitz_part1(x):
    """
    Performs the first part of the Toeplitz matrix multiplication algorithm, which is a Fourier
    transform of the vector x extended
    """
    assert is_power_of_two(len(x))
    
    root_of_unity = get_root_of_unity(len(x) * 2)
    
    # Extend x with zeros (neutral element of G1)
    xext = x + [P1_INF.dup() for _ in range(len(x))]

    xext_fft = fft(xext, MODULUS, root_of_unity, inv=False)
    
    return xext_fft

def toeplitz_part2(toeplitz_coefficients, xext_fft):
    """
    Performs the second part of the Toeplitz matrix multiplication algorithm
    """
    # Extend the toeplitz coefficients to get a circulant matrix into which the Toeplitz
    # matrix is embedded
    assert is_power_of_two(len(toeplitz_coefficients))
    
    root_of_unity = get_root_of_unity(len(xext_fft))

    toeplitz_coefficients_fft = fft(toeplitz_coefficients, MODULUS, root_of_unity, inv=False)
    hext_fft = [v.dup().mult(w) for v, w in zip(xext_fft, toeplitz_coefficients_fft)]

    return hext_fft

def toeplitz_part3(hext_fft):
    root_of_unity = get_root_of_unity(len(hext_fft))

    # Transform back and return the first half of the vector
    # Only the top half is the Toeplitz product, the rest is padding
    return fft(hext_fft, MODULUS, root_of_unity, inv=True)[:len(hext_fft) // 2]


def fk20_single(polynomial, setup):
    """
    Compute all n (single) proofs according to FK20 method
    """

    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    
    x = setup[0][n - 2::-1] + [P1_INF.dup()]
    xext_fft = toeplitz_part1(x)
    
    toeplitz_coefficients = polynomial[-1::] + [0] * (n + 1) + polynomial[1:-1]

    # Compute the vector h from the paper using a Toeplitz matric multiplication
    h = toeplitz_part3(toeplitz_part2(toeplitz_coefficients, xext_fft))

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(n))


# Compute all n (single) proofs according to FK20 method
def fk20_single_data_availability_optimized(polynomial: list[int], setup: tuple[list[blst.P1], list[blst.P2]]) -> list[blst.P1]:
    """
    Special version of the FK20 for the situation of data availability checks:
    The upper half of the polynomial coefficients is always 0, so we do not need to extend to twice the size
    for Toeplitz matrix multiplication
    """
    assert is_power_of_two(len(polynomial))
    
    n = len(polynomial) // 2
    
    assert all(x == 0 for x in polynomial[n:])
    reduced_polynomial = polynomial[:n]
    
    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    x = setup[0][n - 2::-1] + [P1_INF]
    xext_fft = toeplitz_part1(x)
    
    toeplitz_coefficients = reduced_polynomial[-1::] + [0] * (n + 1) + reduced_polynomial[1:-1]

    # Compute the vector h from the paper using a Toeplitz matric multiplication
    h = toeplitz_part3(toeplitz_part2(toeplitz_coefficients, xext_fft))
    
    h = h + [P1_INF.dup() for _ in range(n)] 

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(2 * n))


def data_availabilty_using_fk20(polynomial: list[int], setup: tuple[list[blst.P1], list[blst.P2]]) -> list[blst.P1]:
    """
    Computes all the KZG proofs for data availability checks. This involves sampling on the double domain
    and reordering according to reverse bit order
    """
    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    extended_polynomial = polynomial + [0] * n

    all_proofs = fk20_single_data_availability_optimized(extended_polynomial, setup)

    return list_to_reverse_bit_order(all_proofs)


if __name__ == "__main__":
    #uncomment to report time for the functions
    #from timer import chrono
    #generate_setup = chrono(generate_setup)
    #commit_to_poly = chrono(commit_to_poly)
    #eval_poly_at = chrono(eval_poly_at)
    #check_proof_single = chrono(check_proof_single)
    
    import random
    from tqdm import tqdm

    MAX_DEGREE_POLY = MODULUS-1
    N_POINTS = 512

    polynomial = [random.randint(1, MAX_DEGREE_POLY) for _ in range(512)] 
    n = N_POINTS

    setup = generate_setup(random.getrandbits(256), n)

    commitment = commit_to_poly(polynomial, setup)

    # Computing the proofs on the double 
    all_proofs = data_availabilty_using_fk20(polynomial, setup)
    print("All KZG proofs computed")

    # Now check a random position

    pos = random.randrange(0, len(polynomial), desc='Single point check')
    for pos in tqdm(range(n)):
        root_of_unity = get_root_of_unity(n * 2)
        x = pow(root_of_unity, pos, MODULUS)
        y = eval_poly_at(polynomial, x)
        
        assert check_proof_single(commitment, all_proofs[reverse_bit_order(pos, 2 * n)], x, y, setup)
    print(f"Single point check passed for all {n} points")