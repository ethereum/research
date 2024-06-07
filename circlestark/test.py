from fields import S, M, B, ES, EM, EB
from stark import prove_low_degree, verify_low_degree
from fft import fft, inv_fft
from fast_fft import fft as f_fft, inv_fft as f_inv_fft, np
import time

def test_basic_arithmetic():
    assert (S(12) + S(25)) * S(9) == S(12) * S(9) + S(25) * S(9)
    assert ES([1,2,3,4]) ** (31**4-1) == ES([1,0,0,0])
    print("Basic arithmetic test passed")

def test_fft():
    INPUT_SIZE = 512
    data = [M(3**i) for i in range(INPUT_SIZE)]
    coeffs = fft(data)
    data2 = inv_fft(coeffs)
    assert data2 == data
    print("Basic FFT test passed")

def test_fri():
    print("Testing FRI")
    INPUT_SIZE = 4096
    coeffs = [M(3**i) for i in range(INPUT_SIZE)] + [M(0)] * INPUT_SIZE
    evaluations = inv_fft(coeffs)
    proof = prove_low_degree([EM(v) for v in evaluations])
    length = (
        sum(32 * len(branch) for branch in sum(proof["branches"], []))
        + 32 * len(proof["roots"])
        + 32 * len(proof["leaf_values"])
        + 16 * len(proof["final_values"])
    )
    print("Proof length: {} bytes".format(length))
    
    assert verify_low_degree(proof)
    print("Verified proof")

def test_fast_fft():
    print("Testing fast FFT")
    INPUT_SIZE = 16384
    data = [pow(3, i, 2**31-1) for i in range(INPUT_SIZE)]
    t0 = time.time()
    coeffs1 = fft([B(x) for x in data])
    t1 = time.time()
    print("Computed size-{} slow fft in {} sec".format(INPUT_SIZE, t1 - t0))
    t1 = time.time()
    coeffs2 = f_fft(data)
    t2 = time.time()
    print("Computed size-{} fast fft in {} sec".format(INPUT_SIZE, t2 - t1))
    assert [int(x) for x in coeffs2] == coeffs1
    assert [x for x in f_inv_fft(coeffs2)] == data
    print("Fast FFT checks passed")

if __name__ == '__main__':
    test_basic_arithmetic()
    test_fft()
    test_fri()
    test_fast_fft()
