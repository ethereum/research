import sys
from fields import S, M, B, ES, EM, EB
from fft import fft, inv_fft, log2
from fri import prove_low_degree, verify_low_degree
from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like
)
from utils import (
    pad_to, mk_junk_data
)

from precomputes import sub_domains

from fast_fri import (
    prove_low_degree as f_prove_low_degree,
    verify_low_degree as f_verify_low_degree
)

from line_functions import (
    line_function, interpolant, public_args_to_vanish_and_interp
)

from fast_stark import (
    mk_stark, verify_stark, get_vk, build_constants_tree
)

from fast_fft import (
    fft as f_fft, inv_fft as f_inv_fft, bary_eval
)

from poseidon import (
    poseidon_constraint_check, fill_poseidon_trace, poseidon_hash
)
import cupy as cp

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
    npdata = M31(data)
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
    assert f_inv_fft(coeffs2) == npdata
    print("Fast FFT checks passed")

def test_fast_fri():
    print("Testing FRI")
    INPUT_SIZE = 4096
    coeffs = M31(
        [pow(3, i, modulus) for i in range(INPUT_SIZE)] + [0] * INPUT_SIZE,
    )
    evaluations = f_inv_fft(coeffs)
    print('ev', evaluations, evaluations.__class__)
    proof = f_prove_low_degree(evaluations.to_extended())
    assert f_verify_low_degree(proof)
    global fri_proof
    if fri_proof is None:
        raise Exception("Need slow fri proof to check against")
    assert fri_proof["roots"] == proof["roots"]
    assert fri_proof["branches"] == proof["branches"]
    for leaf_set1, leaf_set2 in zip(fri_proof["leaf_values"], proof["leaf_values"]):
        for leaf1, leaf2 in zip(leaf_set1, leaf_set2):
            for v1, v2 in zip(leaf1, leaf2):
                assert [x.value for x in v1.value] == [int(x) for x in v2.value]
    for v1, v2 in zip(fri_proof["final_values"], proof["final_values"]):
        assert [x.value for x in v1.value] == [int(x) for x in v2.value]
    print("Proofs equivalent!")

def test_mega_fri():
    print("Testing FRI")
    INPUT_SIZE = 2**22
    coeffs = M31.append(
        mk_junk_data(INPUT_SIZE),
        M31.zeros(INPUT_SIZE)
    )
    t1 = time.time()
    evaluations = f_inv_fft(coeffs)
    print("Low-degree extended coeffs in time {}".format(time.time() - t1))
    t2 = time.time()
    proof = f_prove_low_degree(evaluations.to_extended())
    print("Generated proof in time {}".format(time.time() - t2))

def test_lines_and_interpolants():
    coords = (
        # pos1, val1, pos2, val2
        (12, 9, 13, 81, False),
        (8, 16, 15, 256, False),
        (12, [9, 25, 49], 13, [81, 625, 2401], False),
        (8, [999, 998, 997], 15, [900, 901, 902], False),
        (12, [1,2,3,4], 13, [8,16,32,64], True),
        (8, [9,10,11,12], 15, [3,9,27,81], True),
        (8, [[1,2,3,4],[5,0,0,0]], 17, [[8,16,32,64],[900,0,0,0]], True),
        (23, [[5,6,7,80],[9,0,0,0]], 29, [[4,40,14,41],[99,0,0,0]], True),
    )
    for pos1, v1, pos2, v2, is_extended in coords:
        print(pos1, v1, pos2, v2)
        p1 = sub_domains[pos1]
        p2 = sub_domains[pos2]
        p3 = sub_domains[pos2 * 2 - pos1]
        if is_extended:
            p1 = p1.to_extended()
            p2 = p2.to_extended()
            p3 = p3.to_extended()
        v1, v2 = M31(v1), M31(v2)
        L = line_function(p1, p2, sub_domains[8:16])
        assert bary_eval(L, p1) == 0
        assert bary_eval(L, p2) == 0
        assert bary_eval(L, p3) != 0
        I = interpolant(p1, v1, p2, v2, sub_domains[8:16])
        assert bary_eval(I, p1) == v1
        assert bary_eval(I, p2) == v2

    print("Basic line and interpolant checks passed")

    for i in range(0, len(coords), 2):
        pos1, v1, pos2, v2, is_extended = coords[i]
        pos3, v3, pos4, v4, _ = coords[i+1]
        print(pos1, pos2, pos3, pos4, v1, v2, v3, v4)
        cls = ExtendedM31 if is_extended else M31
        V, I = public_args_to_vanish_and_interp(
            32,
            (pos1, pos2, pos3, pos4),
            cls([v1, v2, v3, v4])
        )
        for pos, v in zip((pos1, pos2, pos3, pos4), (v1, v2, v3, v4)):
            assert bary_eval(V, sub_domains[32+pos]) == 0
            assert bary_eval(I, sub_domains[32+pos]) == cls(v)

    print("Vanish-and-interp test passed")

def test_mk_stark():
    def get_next_state(state, constants): 
        o = zeros_like(state)
        o[0] = (state[1] * state[2]) + constants[0]
        o[1] = (state[2] * state[0]) + constants[1]
        o[2] = (state[0] * state[1]) + constants[2]
        return o

    def check_constraint(state, next_state, constants):
        computed_next = get_next_state(state, constants)
        return computed_next - next_state

    trace = M31.zeros((128, 3))
    constants = M31(cp.arange(384, dtype=cp.uint32).reshape((128,3)))

    trace[0] = M31([3, 0, 0])
    for i in range(127):
        trace[i+1] = get_next_state(trace[i], constants[i])
    print("Generating STARK")
    stark = mk_stark(check_constraint, trace, constants, public_args=(0,99), H_degree=4)
    vk = get_vk(trace.shape, constants, 3, (0,99), H_degree=4)
    print("Verifying STARK")
    assert verify_stark(check_constraint, vk, trace[cp.array((0,99))], stark)
    print("Verified!")

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
    print(s.getvalue().replace(os.getcwd(), '.')[:10000])

def test_poseidon_stark():
    NUM_HASHES = 16384
    ins = mk_junk_data(NUM_HASHES*8).reshape((NUM_HASHES,8))
    positions = cp.arange(NUM_HASHES, dtype=cp.uint32) % 2
    constants = M31.zeros((NUM_HASHES,1))
    constants[::32,:] = 1
    k_tree = build_constants_tree(constants, H_degree=4)
    print("Generating Poseidon STARK")
    start_profile()
    trace = fill_poseidon_trace(
        ins,
        positions,
    )
    stark = mk_stark(
        poseidon_constraint_check,
        trace,
        constants,
        public_args=(0,NUM_HASHES-2),
        prebuilt_constants_tree=k_tree,
        H_degree=4
    )
    print(f"Generated proof of {NUM_HASHES-2} hashes")
    end_profile()
    vk = get_vk(trace.shape, constants, 184, (0,NUM_HASHES-2), H_degree=4)

    print("Verifying Poseidon STARK")
    assert verify_stark(
        poseidon_constraint_check,
        vk,
        trace[cp.array((0,NUM_HASHES-2))],
        stark
    )
    print("Verified!")
    fri_proof = stark["fri"]
    L1 = sum(32 * len(branch) for branch in sum(fri_proof["branches"], []))
    L2 = 32 * len(fri_proof["roots"])
    L3 = 32 * len(fri_proof["leaf_values"])
    L4 = 16 * len(fri_proof["final_values"])
    L5 = sum(32 * len(branch) for branch in stark["TQ_branches"])
    L6 = sum(32 * len(branch) for branch in stark["TQ_next_branches"])
    L7 = sum(32 * len(branch) for branch in stark["K_branches"])
    L8 = len(stark["TQ_leaves"].tobytes())
    L9 = len(stark["TQ_next_leaves"].tobytes())
    L10 = len(stark["K_leaves"].tobytes())
    length = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9 + L10
    print(f"Proof length: {length} bytes: {L1} (FRI branches), "
          f"{L2} (FRI roots), {L3} (FRI leaves), {L4} (FRI final values), "
          f"{L5} (T+A branches) {L6} (T+A next branches) {L7} (K branches), "
          f"{L8} (T+A leaves) {L9} (T+A next leaves) {L10} (K leaves)")

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
        test_lines_and_interpolants()
        test_mk_stark()
        test_poseidon_stark()
