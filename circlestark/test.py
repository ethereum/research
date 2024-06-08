from fields import S, M, B, ES, EM, EB
from fft import fft, inv_fft, log2
from stark import prove_low_degree, verify_low_degree
from fast_fft import fft as f_fft, inv_fft as f_inv_fft, np, M31
from fast_stark import (
    prove_low_degree as f_prove_low_degree, to_extension_field
)
import time

def test_basic_arithmetic():
    assert (S(12) + S(25)) * S(9) == S(12) * S(9) + S(25) * S(9)
    assert ES([1,2,3,4]) ** (31**4-1) == ES([1,0,0,0])
    print("Basic arithmetic test passed")

fri_proof = None

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
    coeffs = [B(3**i) for i in range(INPUT_SIZE)] + [B(0)] * INPUT_SIZE
    evaluations = inv_fft(coeffs)
    global fri_proof
    fri_proof = prove_low_degree([EB(v) for v in evaluations])
    length = (
        sum(32 * len(branch) for branch in sum(fri_proof["branches"], []))
        + 32 * len(fri_proof["roots"])
        + 32 * len(fri_proof["leaf_values"])
        + 16 * len(fri_proof["final_values"])
    )
    print("Proof length: {} bytes".format(length))
    
    assert verify_low_degree(fri_proof)
    print("Verified proof")

def test_fast_fft():
    print("Testing fast FFT")
    INPUT_SIZE = 2**13
    data = [pow(3, i, 2**31-1) for i in range(INPUT_SIZE)]
    npdata = np.array(data, dtype=np.uint64)
    t0 = time.time()
    coeffs1 = fft([B(x) for x in data])
    t1 = time.time()
    print("Computed size-{} slow fft in {} sec".format(INPUT_SIZE, t1 - t0))
    t1 = time.time()
    coeffs2 = f_fft(npdata)
    t2 = time.time()
    print("Computed size-{} fast fft in {} sec".format(INPUT_SIZE, t2 - t1))
    assert [int(x) for x in coeffs2] == coeffs1
    assert np.array_equal(f_inv_fft(coeffs2), npdata)
    print("Fast FFT checks passed")

def test_fast_fri():
    print("Testing FRI")
    INPUT_SIZE = 4096
    coeffs = np.array(
        [(3**i) % M31 for i in range(INPUT_SIZE)] + [0] * INPUT_SIZE,
        dtype=np.uint64
    )
    evaluations = f_inv_fft(coeffs)
    proof = f_prove_low_degree(to_extension_field(evaluations))
    global fri_proof
    if fri_proof is None:
        raise Exception("Need slow fri proof to check against")
    assert fri_proof["roots"] == proof["roots"]
    assert fri_proof["branches"] == proof["branches"]
    for leaf_set1, leaf_set2 in zip(fri_proof["leaf_values"], proof["leaf_values"]):
        for leaf1, leaf2 in zip(leaf_set1, leaf_set2):
            for v1, v2 in zip(leaf1, leaf2):
                assert [x.value for x in v1.value] == [int(x) for x in v2]
    for v1, v2 in zip(fri_proof["final_values"], proof["final_values"]):
        assert [x.value for x in v1.value] == [int(x) for x in v2]
    print("Proofs equivalent!")

def test_mega_fri():
    print("Testing FRI")
    INPUT_SIZE = 2**20
    coeffs = np.zeros(INPUT_SIZE * 2, dtype=np.uint64)
    coeffs[:INPUT_SIZE] = 1
    coeffs[1] = 3
    power_of_3 = 3
    for i in range(1, log2(INPUT_SIZE)):
        coeffs[2**i] == (coeffs[2**(i-1)]**2) % M31
    for i in range(1, log2(INPUT_SIZE)):
        coeffs[2**i+1:2**(i+1)] = (coeffs[1:2**i] * coeffs[2**i]) % M31
    t1 = time.time()
    evaluations = f_inv_fft(coeffs)
    print("Low-degree extended coeffs in time {}".format(time.time() - t1))
    t2 = time.time()
    proof = f_prove_low_degree(to_extension_field(evaluations))
    print("Generated proof in time {}".format(time.time() - t2))

if __name__ == '__main__':
    test_basic_arithmetic()
    test_fft()
    test_fri()
    test_fast_fft()
    test_fast_fri()
    test_mega_fri()
