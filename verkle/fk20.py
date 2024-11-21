# Implement technique from https://github.com/khovratovich/Kate/blob/master/Kate_amortized.pdf

from py_ecc import optimized_bls12_381 as b
from fft import fft
from poly_utils import PrimeField
from multicombs import lincomb
from verkle import WIDTH, DEPTH, MODULUS, root_of_unity_candidates, ROOT_OF_UNITY
import time

ROOT_OF_UNITY2 = root_of_unity_candidates[WIDTH*2]


# default O(n^2) algorithm
def semi_toeplitz_naive(toeplitz_coefficients, x):
    r = []
    for i in range(len(x)):
        r.append(sum(toeplitz_coefficients[j - i] * x[j] for j in range(i, len(x))))
    return r


xext_hat = None

# FFT algorithm
# Toeplitz multiplication via http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html
def semi_toeplitz_fft(toeplitz_coefficients, x):
    global xext_hat
    assert len(x) == WIDTH
    if xext_hat == None:
        a = time.time()
        if type(x[0]) == tuple:
            xext = x + [b.Z1 for a in x]
        else:
            xext = x + [0 * a for a in x]

        xext_hat = fft(xext, MODULUS, ROOT_OF_UNITY2, inv=False)
        print("Toeplitz preprocessing in %.3f seconds" % (time.time() - a))

    text = toeplitz_coefficients[:1] + [0 * a for a in toeplitz_coefficients] + toeplitz_coefficients[:0:-1]
    text_hat = fft(text, MODULUS, ROOT_OF_UNITY2, inv=False)
    yext_hat = [None for i in range(2*len(x))]
    for i in range(len(xext_hat)):
        if type(xext_hat[0]) == tuple:
            yext_hat[i] = b.multiply(xext_hat[i], text_hat[i])
        else:
            yext_hat[i] *= text_hat[i]
    return fft(yext_hat, MODULUS, ROOT_OF_UNITY2, inv=True)[:len(x)]

def generate_all_proofs(values, setup):
    assert len(values) == WIDTH
    # Get polynomial coefficients using IFFT
    print("---")
    print("Generating all proofs using FK20 for width = %d" % WIDTH)
    a = time.time()
    coefs = fft(values, MODULUS, ROOT_OF_UNITY, inv=True)[:0:-1]
    print("Generated polynomial coefficients in %.3f seconds" % (time.time() - a))

    a = time.time()
    h = semi_toeplitz_fft(coefs + [0], setup[0][len(values)-2::-1] + [b.Z1])
    print("Toeplitz matrix multiplication in %.3f seconds" % (time.time() - a))

    a = time.time()
    r = fft(h, MODULUS, ROOT_OF_UNITY)
    print("Final FFT in %.3f seconds" % (time.time() - a))
    print("---")

    return r