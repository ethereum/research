from py_ecc import optimized_bls12_381 as b
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
    eval_poly_at
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


# The componation toeplitz_part2(toeplitz_coefficients, toeplitz_part1(x)) compute the matrix-vector
# multiplication
# T * x
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
    xext = x + [b.Z1] * len(x)

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
    yext_fft = [b.multiply(v, w) for v, w in zip(xext_fft, toeplitz_coefficients_fft)]

    # Transform back and return the first half of the vector
    # Only the top half is the Toeplitz product, the rest is padding
    return fft(yext_fft, MODULUS, root_of_unity, inv=True)[:len(yext_fft) // 2]


def fk20_single(polynomial):
    """
    Compute all n (single) proofs according to FK20 method
    """

    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    
    x = setup[0][n - 2::-1] + [b.Z1]
    xext_fft = toeplitz_part1(x)
    
    toeplitz_coefficients = polynomial[-1::] + [0] * (n + 1) + polynomial[1:-1]

    # Compute the vector h from the paper using a Toeplitz matric multiplication
    h = toeplitz_part2(toeplitz_coefficients, xext_fft)

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(n))


def fk20_data_availabilty(polynomial):
    """
    Computes all the KZG proofs for data availability checks. This involves sampling on the double domain
    and reordering according to reverse bit order
    """
    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    extended_polynomial = polynomial + [0] * n

    all_proofs = fk20_single(extended_polynomial)

    return list_to_reverse_bit_order(all_proofs)


if __name__ == "__main__":
    setup = generate_setup(1927409816240961209460912649124)
    kzg_proofs.setup = setup

    polynomial = [1, 2, 3, 4, 7, 7, 7, 7, 13, 13, 13, 13, 13, 13, 13, 13]
    n = len(polynomial)
    commitment = commit_to_poly(polynomial)

    # Computing the proofs on the double 
    all_proofs = fk20_data_availabilty(polynomial)
    print("All KZG proofs computed")

    # Now check a random position

    pos = 9
    root_of_unity = get_root_of_unity(n * 2)
    x = pow(root_of_unity, pos, MODULUS)
    y = eval_poly_at(polynomial, x)
    
    assert check_proof_single(commitment, all_proofs[reverse_bit_order(pos, 2 * n)], x, y)
    print("Single point check passed")