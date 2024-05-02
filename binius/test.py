from binary_fields import BinaryFieldElement as B
from utils import extend, log2, eval_poly_at, multilinear_poly_eval
from simple_binius import simple_binius_proof, verify_simple_binius_proof
from packed_binius import (
    packed_binius_proof, verify_packed_binius_proof
)
from binary_ntt import (
    additive_ntt, inv_additive_ntt, eval_poly_in_basis, get_Wi_eval, get_Wi
)
from crazy_ntt import (
    int_to_bigbin, bigbin_to_int, Wi_eval_cache,
    multilinear_poly_eval as crazy_ml_eval, big_mul
)
import numpy as np

def compute_size(x):
    if isinstance(x, bytes):
        return len(x)
    elif isinstance(x, B):
        return (x.bit_length() + 7) // 8
    elif isinstance(x, list):
        return sum(compute_size(val) for val in x)
    elif isinstance(x, dict):
        return sum(compute_size(val) for val in x.values())
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
        get_Wi_eval(7, 383) ==
        eval_poly_at(get_Wi(7), B(383))
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
    evals = [3, 14, 15, B(92)]
    assert crazy_ml_eval(evals, [0, 0]) == 3
    assert crazy_ml_eval(evals, [1, 0]) == 14
    assert crazy_ml_eval(evals, [2, 5]) == 204
    SIZE = 256
    z = [B(int(bit)) for bit in bin(3**SIZE)[2:][:SIZE]]
    eval_point = [B((999**i)%2**128) for i in range(log2(SIZE))]
    assert crazy_ml_eval(z, eval_point) == multilinear_poly_eval(z, eval_point)
    print("Verified vectorized operations")


def test_simple_binius():
    SIZE = 16384
    z = [B(int(bit)) for bit in bin(3**SIZE)[2:][:SIZE]]
    eval_point = [B((999**i)%2**128) for i in range(log2(SIZE))]
    proof = simple_binius_proof(z, eval_point)
    print("Generated simple-binius proof")
    print("Proof size: {} bytes".format(compute_size(proof)))
    # print("t_prime:", proof["t_prime"])
    verify_simple_binius_proof(proof)
    print("Verified simple-binius proof")

def test_packed_binius():
    SIZE = 2**20
    z = [B(int(bit)) for bit in bin(3**SIZE)[2:][:SIZE]]
    eval_point = [B((999**i)%2**128) for i in range(log2(SIZE))]
    proof = packed_binius_proof(z, eval_point)
    print("Generated packed-binius proof")
    print("Proof size: {} bytes".format(compute_size(proof)))
    # print("t_prime:", proof["t_prime"])
    verify_packed_binius_proof(proof)
    print("Verified packed-binius proof")

if __name__ == '__main__':
    test_binary_operations()
    test_vectorized_operations()
    test_simple_binius()
    test_packed_binius()
