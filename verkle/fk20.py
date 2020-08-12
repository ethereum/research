# Implement technique from https://github.com/khovratovich/Kate/blob/master/Kate_amortized.pdf

from py_ecc import optimized_bls12_381 as b
from fft import fft
from poly_utils import PrimeField
from multicombs import lincomb
from verkle import WIDTH, DEPTH, MODULUS, root_of_unity_candidates, ROOT_OF_UNITY

ROOT_OF_UNITY2 = root_of_unity_candidates[WIDTH*2]


# default O(n^2) algorithm
def semi_toeplitz_naive(toeplitz_coefficients, x):
    r = []
    for i in range(len(x)):
        r.append(sum(toeplitz_coefficients[j - i] * x[j] for j in range(i, len(x))))
    return r


# FFT algorithm
# Toeplitz multiplication via http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html
def semi_toeplitz_fft(toeplitz_coefficients, x):
    assert len(x) == WIDTH
    if type(x[0]) == tuple:
        xext = x + [b.Z1 for a in x]
    else:
        xext = x + [0 * a for a in x]
    xext_hat = fft(xext, MODULUS, ROOT_OF_UNITY2, inv=False)
    text = toeplitz_coefficients[:1] + [0 * a for a in toeplitz_coefficients] + toeplitz_coefficients[:0:-1]
    text_hat = fft(text, MODULUS, ROOT_OF_UNITY2, inv=False)
    for i in range(len(xext_hat)):
        if type(xext_hat[0]) == tuple:
            xext_hat[i] = b.multiply(xext_hat[i], text_hat[i])
        else:
            xext_hat[i] *= text_hat[i]
    return fft(xext_hat, MODULUS, ROOT_OF_UNITY2, inv=True)[:len(x)]

def generate_all_proofs(values, setup):
    assert len(values) == WIDTH
    # Get polynomial coefficients using IFFT
    coefs = fft(values, MODULUS, ROOT_OF_UNITY, inv=True)[:0:-1]
    h = semi_toeplitz_fft(coefs + [0], setup[0][len(values)-2::-1] + [b.Z1])
    return fft(h, MODULUS, ROOT_OF_UNITY)