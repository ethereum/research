import sys
from fields import S, M, B, ES, EM, EB
from fft import fft, inv_fft, log2
from fri import prove_low_degree, verify_low_degree
from utils import (
    np, M31, modinv, to_extension_field, zeros, arange, array,
    append, pad_to, m31_arith, ext_arith, mk_junk_data
)

from precomputes import sub_domains

from fast_fft import (
    fft as f_fft, inv_fft as f_inv_fft, bary_eval
)

from fast_fri import (
    prove_low_degree as f_prove_low_degree,
    verify_low_degree as f_verify_low_degree
)

from line_functions import (
    line_function, interpolant, public_args_to_vanish_and_interp
)

from fast_stark import (
    mk_stark, verify_stark, get_vk,
)

from arithmetization_builder import (
    example, example_args, generate_constants_table, generate_arguments_table,
    generate_filled_trace, generate_next_state_function,
    get_public_args_indices, get_arguments_width
)

from poseidon import (
    poseidon_hash, arith_hash, arith_hash2, poseidon_next_state,
    poseidon_constants, poseidon_hasher, poseidon_branch_hasher, NUM_HASHES
)

import time
import cProfile
import pstats
import io
import os

def test_basic_arithmetic():
    assert (S(12) + S(25)) * S(9) == S(12) * S(9) + S(25) * S(9)
    assert ES([1,2,3,4]) ** (31**4-1) == ES([1,0,0,0])
    print("Basic arithmetic test passed")

fri_proof = None
test_stark_output = None

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
    npdata = array(data)
    t0 = time.time()
    coeffs1 = fft([B(x) for x in data])
    t1 = time.time()
    print("Computed size-{} slow fft in {} sec".format(INPUT_SIZE, t1 - t0))
    t1 = time.time()
    coeffs2 = f_fft(npdata)
    print(coeffs2)
    t2 = time.time()
    print("Computed size-{} fast fft in {} sec".format(INPUT_SIZE, t2 - t1))
    assert [int(x) for x in coeffs2] == coeffs1
    assert np.array_equal(f_inv_fft(coeffs2), npdata)
    print("Fast FFT checks passed")

def test_fast_fri():
    print("Testing FRI")
    INPUT_SIZE = 4096
    coeffs = array(
        [pow(3, i, M31) for i in range(INPUT_SIZE)] + [0] * INPUT_SIZE,
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
    INPUT_SIZE = 2**22
    coeffs = append(
        mk_junk_data(INPUT_SIZE),
        zeros(INPUT_SIZE)
    )
    t1 = time.time()
    evaluations = f_inv_fft(coeffs)
    print("Low-degree extended coeffs in time {}".format(time.time() - t1))
    t2 = time.time()
    proof = f_prove_low_degree(to_extension_field(evaluations))
    print("Generated proof in time {}".format(time.time() - t2))

def test_simple_arithmetize():
    print("Testing simple arithmetization")
    SIZE = 128
    trace = zeros(SIZE)
    trace[0] = 1
    for i in range(SIZE-1):
        trace[i+1] = ((trace[i] ** 2 % M31) * trace[i] + 1) % M31
    ext_trace = f_inv_fft(pad_to(f_fft(trace), SIZE*4))
    assert bary_eval(ext_trace, sub_domains[SIZE], m31_arith) == 1
    assert bary_eval(ext_trace, sub_domains[SIZE+1], m31_arith) == 2
    assert bary_eval(ext_trace, sub_domains[SIZE+2], m31_arith) == 9
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
        zeros(SIZE-1)
    )
    print("Simple arithmetization test passed")

def test_lines_and_interpolants():
    coords = (
        # pos1, val1, pos2, val2
        (12, 9, 13, 81, m31_arith),
        (8, 16, 15, 256, m31_arith),
        (12, [9, 25, 49], 13, [81, 625, 2401], m31_arith),
        (8, [999, 998, 997], 15, [900, 901, 902], m31_arith),
        (12, [1,2,3,4], 13, [8,16,32,64], ext_arith),
        (8, [9,10,11,12], 15, [3,9,27,81], ext_arith),
        (8, [[1,2,3,4],[5,0,0,0]], 17, [[8,16,32,64],[900,0,0,0]], ext_arith),
        (23, [[5,6,7,80],[9,0,0,0]], 29, [[4,40,14,41],[99,0,0,0]], ext_arith),
    )
    for pos1, v1, pos2, v2, arith in coords:
        print(pos1, v1, pos2, v2)
        p1 = sub_domains[pos1]
        p2 = sub_domains[pos2]
        p3 = sub_domains[pos2 * 2 - pos1]
        if arith[0].ndim == 1:
            p1 = to_extension_field(p1)
            p2 = to_extension_field(p2)
            p3 = to_extension_field(p3)
        v1, v2 = array(v1), array(v2)
        L = line_function(p1, p2, sub_domains[8:16], arith)
        assert not np.any(bary_eval(L, p1, arith))
        assert not np.any(bary_eval(L, p2, arith))
        assert np.any(bary_eval(L, p3, arith))
        I = interpolant(p1, v1, p2, v2, sub_domains[8:16], arith)
        assert np.array_equal(bary_eval(I, p1, arith), v1)
        assert np.array_equal(bary_eval(I, p2, arith), v2)

    print("Basic line and interpolant checks passed")

    for i in range(0, len(coords), 2):
        pos1, v1, pos2, v2, arith = coords[i]
        pos3, v3, pos4, v4, _ = coords[i+1]
        print(pos1, pos2, pos3, pos4, v1, v2, v3, v4)
        V, I = public_args_to_vanish_and_interp(
            32,
            (pos1, pos2, pos3, pos4),
            array([v1, v2, v3, v4]),
            arith
        )
        for pos, v in zip((pos1, pos2, pos3, pos4), (v1, v2, v3, v4)):
            assert not np.any(bary_eval(V, sub_domains[32+pos], arith))
            assert np.array_equal(
                bary_eval(I, sub_domains[32+pos], arith),
                array(v)
            )

    print("Vanish-and-interp test passed")

def test_mk_stark():
    def next_state(state, constants, arguments, arith): 
        one, add, mul = arith
        o = add(
            mul(constants[0], np.stack((
                add(mul(state[1], state[2]), one),
                add(mul(state[2], state[0]), one),
                add(mul(state[0], state[1]), one),
            ))),
            mul(constants[1], add(state, arguments))
        )
        return o

    constants = array([[0,1]] + [[1,0]]*98 + [[0,1]])
    arguments = array([[3,0,0]] + [[0,0,0]] * 99)
    print("Generating STARK")
    stark = mk_stark(next_state, 3, constants, arguments, public_args=(0,99))
    global test_stark_output
    test_stark_output = stark["output_state"]
    vk = get_vk(3, constants, arguments.shape[1], (0,99))
    print("Verifying STARK")
    assert verify_stark(next_state, vk, arguments[array((0,99))], stark)
    print("Verified!")

def test_arithmetization_builder():
    constants = generate_constants_table(example)
    arguments = generate_arguments_table(example, example_args)
    prefilled_trace = generate_filled_trace(example, constants, arguments)
    next_state = generate_next_state_function(example)
    indices = get_public_args_indices(example)
    stark = mk_stark(
        next_state,
        example["trace_width"],
        constants,
        arguments,
        indices,
        prefilled_trace=prefilled_trace
    )
    vk = get_vk(3, constants, arguments.shape[1], indices)
    print("Verifying STARK")
    assert verify_stark(next_state, vk, arguments[array(indices)], stark)
    print("Verified!")
    global test_stark_output
    if test_stark_output is None:
        raise Exception("Need test_mk_stark to check against")
    assert np.array_equal(stark["output_state"], test_stark_output)
    print("Outputs confirmed match")

def start_profile():
    global profiler
    profiler = cProfile.Profile()
    profiler.enable()

def end_profile():
    global profiler
    profiler.disable()
    # Print the profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(s.getvalue().replace(os.getcwd(), '.')[:4000])

def test_poseidon_stark():
    in1 = arange(8)
    in2 = arange(8, 16)
    h1 = poseidon_hash(in1, in2)
    print("Directly computed hash:", h1)
    h2 = arith_hash(in1, in2)
    print("Hash from raw arithmetization:", h2)
    h3 = arith_hash2(in1, in2)
    print("Hash from arithmetization builder:", h3)
    assert np.array_equal(h1, h2)
    assert np.array_equal(h1, h3)
    branch_arguments = {"load_args": [
        append(arange(i*8, (i+1)*8), array([i%2])) for i in range(NUM_HASHES+1)
    ]}
    constants = generate_constants_table(poseidon_branch_hasher)
    positions = get_public_args_indices(poseidon_branch_hasher)
    vk = get_vk(
        poseidon_branch_hasher["trace_width"],
        constants,
        get_arguments_width(poseidon_branch_hasher),
        positions
    )
    print("Generating Poseidon STARK")
    start_profile()
    arguments = generate_arguments_table(
        poseidon_branch_hasher,
        branch_arguments
    )
    prefilled_trace = generate_filled_trace(
        poseidon_branch_hasher,
        constants,
        arguments
    )
    next_state = generate_next_state_function(poseidon_branch_hasher)
    stark = mk_stark(
        poseidon_next_state,
        poseidon_branch_hasher["trace_width"],
        constants,
        arguments,
        positions,
        prefilled_trace=prefilled_trace
    )
    print("Generated")
    end_profile()
    print("Verifying Poseidon STARK")
    assert verify_stark(poseidon_next_state, vk, arguments[array(positions)], stark)
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
