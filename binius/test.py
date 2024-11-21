from binary_fields import BinaryFieldElement as B
from utils import (
    extend, log2, eval_poly_at, multilinear_poly_eval,
    evaluation_tensor_product
)
from simple_binius import simple_binius_proof, verify_simple_binius_proof
from packed_binius import (
    packed_binius_proof, verify_packed_binius_proof
)
from optimized_binius import (
    optimized_binius_proof, verify_optimized_binius_proof,
)
from binary_ntt import (
    additive_ntt, inv_additive_ntt, eval_poly_in_basis, get_Wi_eval, get_Wi
)
from optimized_utils import (
    int_to_bigbin, bigbin_to_int, Wi_eval_cache,
    multilinear_poly_eval as op_multilinear_poly_eval, big_mul, bytestobits,
    evaluation_tensor_product as op_evaluation_tensor_product, multisubset,
    np
)
from hashlib import sha256

# 256 MB
TEST_DATA_STREAM = (3 ** np.arange(2**25, dtype=np.uint64)).tobytes('C')

#TEST_DATA_STREAM = b''.join(sha256(i.to_bytes(4, 'big')).digest() for i in range(2**23))

def compute_size(x):
    if isinstance(x, bytes):
        return len(x)
    elif isinstance(x, B):
        return (x.bit_length() + 7) // 8
    elif isinstance(x, list):
        return sum(compute_size(val) for val in x)
    elif isinstance(x, dict):
        return sum(compute_size(val) for val in x.values())
    elif isinstance(x, np.ndarray):
        return x.size * x.dtype.itemsize
    elif isinstance(x, int):
        assert x < 2**128
        return 16
    else:
        raise Exception("Error computing size of {}".format(x))

def test_binary_operations():
    # Basic arithmetic
    assert (B(3) + B(14)) * B(15) == B(3) * B(15) + B(14) * B(15)
    assert len(set((B(x) * 3).value for x in range(256))) == 256
    assert B(13) ** 255 == 1
    assert B(420) ** 255 != 1
    assert B(420) ** 65535 == 1

    # Polynomial evaluations
    evals = [3, 14, 15, B(92)]
    assert multilinear_poly_eval(evals, [0, 0]) == 3
    assert multilinear_poly_eval(evals, [1, 0]) == 14
    assert multilinear_poly_eval(evals, [2, 5]) == 204

    # Reed-Solomon extension
    poly = [B((3**i)%256) for i in range(8)]
    ntt = additive_ntt(poly)
    assert ntt == [eval_poly_in_basis(poly, i) for i in range(8)]
    assert inv_additive_ntt(ntt) == poly
    print("Verified basic operations")

def test_vectorized_operations():
    # Check that the two Wi cache implementations give correct values
    assert (
        Wi_eval_cache[7, 383] ==
        get_Wi_eval(7, 383).value ==
        eval_poly_at(get_Wi(7), B(383)).value
    )
    # Single value by single value
    assert (
        bigbin_to_int(big_mul(int_to_bigbin(3**29), int_to_bigbin(5**29))) ==
        (B(3**29) * B(5**29)).value
    )
    
    # Vector by vector
    assert (
        [
            bigbin_to_int(x) for x in big_mul(
                np.array([int_to_bigbin(3**i) for i in range(55)]),
                np.array([int_to_bigbin(5**i) for i in range(55)])
            )
        ] == [
            (B(3**i) * B(5**i)).value for i in range(55)
        ]
    )

    # Vector by single value
    assert(
        [
            bigbin_to_int(x) for x in big_mul(
                np.array([int_to_bigbin(3**i) for i in range(55)]),
                int_to_bigbin(123456789)
            )
        ] == [
            (B(3**i) * B(123456789)).value for i in range(55)
        ]
    )

    # Polynomial evaluation consistency
    evals = bytes([3, 14, 15, 92]) # 11000000 01110000 11110000 00111010
    b_evals = list(B(int(x)) for x in bytestobits(evals))
    assert bigbin_to_int(op_multilinear_poly_eval(evals, [0, 0, 0, 0, 0])) == 1
    assert bigbin_to_int(op_multilinear_poly_eval(evals, [1, 0, 0, 0, 0])) == 1
    assert bigbin_to_int(op_multilinear_poly_eval(evals, [0, 1, 0, 0, 0])) == 0
    pt = [9999**i for i in range(5)]
    b_pt = [B(x) for x in pt]
    control_value = multilinear_poly_eval(b_evals, b_pt)
    assert bigbin_to_int(op_multilinear_poly_eval(evals, pt)) == control_value
    print("Verified vectorized operations")

    # Evaluation tensor product consistency
    etp = evaluation_tensor_product(b_pt)
    op_etp = op_evaluation_tensor_product(pt)
    assert [bigbin_to_int(x) for x in op_etp] == etp

    # Multi-subset. Make sure that the method works for different
    # dimensionalities both of inputs and of bits
    values = np.array([1<<i for i in range(16)], dtype=np.uint16)
    bits = np.array(
        [
            [((3**j) >> i) & 1 for i in range(16)]
            for j in range(8)
        ], dtype=np.uint16
    )
    answer = np.array([3**i for i in range(8)], dtype=np.uint16)
    assert np.array_equal(multisubset(values, bits), answer)
    assert np.array_equal(
        multisubset(values, bits.reshape(4, 2, 16)),
        answer.reshape(4, 2)
    )
    values2 = np.transpose(np.array([values, values*2]))
    answer2 = np.array([[3**i, (3**i)*2] for i in range(8)], dtype=np.uint16)
    assert np.array_equal(multisubset(values2, bits), answer2)
    assert np.array_equal(
        multisubset(values2, bits.reshape(4, 2, 16)),
        answer2.reshape(4, 2, 2)
    )


def test_simple_binius(size):
    z = [B(int(byte)) for byte in TEST_DATA_STREAM[:size]]
    eval_point = [B((999**i)%2**128) for i in range(log2(size))]
    proof = simple_binius_proof(z, eval_point)
    print("Generated simple-binius proof")
    print("Proof size: {} bytes".format(compute_size(proof)))
    # print("t_prime:", proof["t_prime"])
    verify_simple_binius_proof(proof)
    print("Verified simple-binius proof")

def test_packed_binius(size):
    z = [B((int(TEST_DATA_STREAM[i//8]) >> (i%8)) & 1) for i in range(size)]
    eval_point = [B((999**i)%2**128) for i in range(log2(size))]
    proof = packed_binius_proof(z, eval_point)
    print("Generated packed-binius proof")
    print("Proof size: {} bytes".format(compute_size(proof)))
    # print("t_prime:", proof["t_prime"])
    verify_packed_binius_proof(proof)
    print("Verified packed-binius proof")

def test_optimized_binius(size):
    z = TEST_DATA_STREAM[:size//8]
    eval_point = [(999**i)%2**128 for i in range(log2(size))]
    proof = optimized_binius_proof(z, eval_point)
    print("Generated packed-binius proof")
    print("Proof size: {} bytes".format(compute_size(proof)))
    # print("t_prime:", proof["t_prime"])
    verify_optimized_binius_proof(proof)
    print("Verified optimized-binius proof")

if __name__ == '__main__':
    test_binary_operations()
    test_vectorized_operations()
    test_simple_binius(2**14)
    test_packed_binius(2**17)
    test_optimized_binius(2**31)
