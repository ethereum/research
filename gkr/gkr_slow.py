from zorch import koalabear
from utils import hash
from sumcheck import sumcheck_prove, sumcheck_verify
import time
from gkr_utils import (
    M, matmul_layer, chi_weights, mle_eval, chi_eval, compute_weights,
    fast_point_eval, generate_weights
)

def compute_permutation(values):
    for i in range(64):
        values = matmul_layer(values)**3 + i
    return values

def gkr_prove(values):
    layers = [values]
    for i in range(64):
        layers.append(matmul_layer(layers[-1]) ** 3 + i)
    randomness = hash(layers[-1], values)
    weights = generate_weights(randomness, values.shape[0])
    proof = []
    Z_top = koalabear.ExtendedKoalaBear.sum(layers[-1] * weights, axis=0)
    Z = Z_top
    for i in range(63, -1, -1):
        # Going in, we have an "obligation" to prove that
        # weights * (V**3 + i) sums to Z, initially Z_top
        V = matmul_layer(layers[i])
        c0, c1, c2, c3, c4, p, W_p, V_p = sumcheck_prove(weights, V, i, randomness)
        # Now, we have an obligation to prove V(p) = V_p
        proof.append((c0, c1, c2, c3, c4, p, V_p))
        Z = V_p
        # V(p) equals the linear combination sum(chi_weights(p) * V). So we
        # compute the coeffs of that linear combination, and we now again
        # have an obligation in the format of the previous layer
        weights = compute_weights(p)
    return Z_top, proof

def gkr_verify(values, outputs, Z_top, proof):
    randomness = hash(outputs, values)
    weights = generate_weights(randomness, values.shape[0])
    assert Z_top == koalabear.ExtendedKoalaBear.sum(outputs * weights, axis=0)
    Z = Z_top
    prev_p = None
    for i, (c0, c1, c2, c3, c4, p, V_p) in zip(range(63, -1, -1), proof):
        W_p = mle_eval(weights, p) if i==63 else fast_point_eval(prev_p, p)
        # The first auxiliary to the obligation reduction is the sumcheck
        sumcheck_verify(c0, c1, c2, c3, c4, Z, p, W_p * (V_p ** 3 + i), randomness)
        Z = V_p
        prev_p = p
        if i == 0:
            assert V_p == mle_eval(matmul_layer(values), p)

    return True


def test():
    coord1 = koalabear.ExtendedKoalaBear([123, 45, 678, 90])
    coord2 = koalabear.ExtendedKoalaBear([2718, 2818, 284590, 45])
    weights = chi_weights(coord1)
    eval1 = mle_eval(weights, coord2)
    eval2 = chi_eval(coord1, coord2)
    assert eval1 == eval2
    print("Chi polynomial tests passed")
    values = koalabear.KoalaBear(list(range(1024)))
    t1 = time.time()
    outputs = compute_permutation(values)
    print("Raw execution time:", time.time() - t1)
    t2 = time.time()
    Z_top, proof = gkr_prove(values)
    print("Proof generated")
    print("Generation time:", time.time() - t2)
    assert gkr_verify(values, outputs, Z_top, proof)
    print("Verification completed")

if __name__ == '__main__':
    test()
