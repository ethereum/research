import sys
from fields import S, M, B, ES, EM, EB
from fft import fft, inv_fft, log2
from fri import prove_low_degree, verify_low_degree
from fast_fft import (
    fft as f_fft, inv_fft as f_inv_fft, np, M31, modinv,
    sub_domains, bary_eval, to_extension_field
)
from fast_fri import (
    prove_low_degree as f_prove_low_degree,
    verify_low_degree as f_verify_low_degree
)
from fast_arithmetize import pad_to, mk_stark, verify_stark, get_vk
from poseidon import (
    poseidon_hash, arith_hash, poseidon_next_state, poseidon_constants
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
    assert f_verify_low_degree(proof)
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

    def mk_junk_data(length):
        a = np.arange(length, length*2, dtype=np.uint64)
        return ((3**a) ^ (7**a)) % M31

    coeffs[:INPUT_SIZE] = mk_junk_data(INPUT_SIZE)
    t1 = time.time()
    evaluations = f_inv_fft(coeffs)
    print("Low-degree extended coeffs in time {}".format(time.time() - t1))
    t2 = time.time()
    proof = f_prove_low_degree(to_extension_field(evaluations))
    print("Generated proof in time {}".format(time.time() - t2))

def test_simple_arithmetize():
    print("Testing simple arithmetization")
    SIZE = 128
    trace = np.zeros(SIZE, dtype=np.uint64)
    trace[0] = 1
    for i in range(SIZE-1):
        trace[i+1] = ((trace[i] ** 2 % M31) * trace[i] + 1) % M31
    ext_trace = f_inv_fft(pad_to(f_fft(trace), SIZE*4))
    assert bary_eval(ext_trace, sub_domains[SIZE]) == 1
    assert bary_eval(ext_trace, sub_domains[SIZE+1]) == 2
    assert bary_eval(ext_trace, sub_domains[SIZE+2]) == 9
    C_left = np.roll(ext_trace, -4)
    C_right = ((ext_trace**2 % M31) * ext_trace + M31 + 1) % M31
    C = (
        np.roll(ext_trace, -4)
        + M31 - ((ext_trace**2 % M31) * ext_trace + M31 + 1) % M31
    ) % M31
    C_exempt_start = C * (
        sub_domains[SIZE*4:SIZE*8, 0] + M31 - sub_domains[SIZE, 0]
    ) % M31
    Z = sub_domains[SIZE*4:SIZE*8, 0]
    for i in range(1, log2(SIZE)):
        Z = (2*Z**2 + M31 - 1) % M31
    assert np.array_equal(
        f_fft(C_exempt_start * modinv(Z) % M31)[SIZE*3+1:],
        np.zeros(SIZE-1, dtype=np.uint64)
    )
    print("Simple arithmetization test passed")

def test_mk_stark():
    def next_state(state, c, arith): 
        one, add, mul = arith
        o = np.array([
            add(mul(mul(state[0], state[0]), state[1]), c[0]),
            add(mul(mul(state[1], state[1]), state[2]), c[0]),
            add(mul(mul(state[2], state[2]), state[0]), c[0], one),
        ])
        return o

    constants = np.arange(100, dtype=np.uint64).reshape((100,1))
    start_state = np.array([1,1,2], dtype=np.uint64)
    print("Generating STARK")
    stark = mk_stark(next_state, start_state, constants)
    vk = get_vk(constants)
    print("Verifying STARK")
    assert verify_stark(next_state, vk, start_state, stark)
    print("Verified!")

def test_poseidon_stark():
    in1 = np.arange(8, dtype=np.uint64)
    in2 = np.arange(8, 16, dtype=np.uint64)
    h1 = poseidon_hash(in1, in2)
    print("Directly computed hash:", h1)
    h2 = arith_hash(in1, in2)
    print("Hash from arithmetization:", h2)
    #assert np.array_equal(h1, h2)
    vk = get_vk(poseidon_constants)
    print("Generating Poseidon STARK")
    start_state = np.append(np.append(in1, in2), np.zeros(32, dtype=np.uint64))
    stark = mk_stark(poseidon_next_state, start_state, poseidon_constants)
    print("Generated")
    print("Verifying Poseidon STARK")
    assert verify_stark(poseidon_next_state, vk, start_state, stark)
    print("Verified!")

if __name__ == '__main__':
    for name, func in dict(locals()).items():
        if name in sys.argv:
            func()
    if len(sys.argv) <= 1:
        test_basic_arithmetic()
        test_fft()
        test_fri()
        test_fast_fft()
        test_fast_fri()
        test_mega_fri()
        test_simple_arithmetize()
        test_mk_stark()
        test_poseidon_stark()
