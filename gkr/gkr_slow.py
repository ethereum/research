from zorch import koalabear
from sumcheck import sumcheck_prove, sumcheck_verify
import time
from gkr_utils import (
    M, matmul_layer, chi_weights, mle_eval, chi_eval, compute_weights,
    fast_point_eval, generate_weights_seed_coords, generate_weights,
    log2, hash as _hash
)

# This file does a simple GKR for a simple version of Poseidon:
# X -> MX ** 3 + i, where M is multiplication by the round matrix
def permutation(values):
    for i in range(64):
        values = matmul_layer(values)**3 + i
    return values

def hash(*inputs):
    return _hash(*inputs, permutation=permutation)

# Prove evaluations via the GKR algorithm
def gkr_prove(values):
    layers = [values]
    post_matmul_layers = []
    for i in range(64):
        post_matmul_layers.append(matmul_layer(layers[-1]))
        layers.append(post_matmul_layers[-1] ** 3 + i)
    randomness = hash(layers[-1], values)
    weights = generate_weights(randomness, values.shape[0], hash)
    proof = []
    Z_top = koalabear.ExtendedKoalaBear.sum(layers[-1] * weights, axis=0)
    Z = Z_top
    for i in range(63, -1, -1):
        # Going in, we have an "obligation" to prove that
        # weights * (V**3 + i) sums to Z, initially Z_top
        V = post_matmul_layers[i]
        c0, c1, c2, c3, c4, p, W_p, V_p = \
                sumcheck_prove(weights, V, 0, i, randomness, hash)
        # Now, we have an obligation to prove V(p) = V_p
        proof.append((c0, c1, c2, c3, c4, p, V_p))
        Z = V_p
        # V(p) equals the linear combination sum(chi_weights(p) * V). So we
        # compute the coeffs of that linear combination, and we now again
        # have an obligation in the format of the previous layer
        weights = compute_weights(p)
        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])
    return Z_top, proof

# Verify a GKR proof
def gkr_verify(values, outputs, Z_top, proof):
    randomness = hash(outputs, values)
    weight_seed_coords = generate_weights_seed_coords(
        randomness,
        values.shape[0],
        hash
    )
    # Verify that the provided Z_top equals sum(outputs * initial_weights)
    assert Z_top == mle_eval(outputs, weight_seed_coords)
    Z = Z_top
    prev_p = None
    # Walk through the layers backwards, and verify each proof
    for i, (c0, c1, c2, c3, c4, p, V_p) in zip(range(63, -1, -1), proof):
        if i == 63:
            W_p = chi_eval(weight_seed_coords, p)
        else:
            W_p = fast_point_eval(prev_p, p)
        # Verify the claim of layer * weights = sum, and reduce it to a claim
        # prev_layer(point) = value. In the next step, we use chi_eval to
        # re-interpret this claim as prev_layer * prev_weights = prev_sum
        sumcheck_verify(
            c0, c1, c2, c3, c4, Z, p, W_p * (V_p ** 3 + i), randomness, hash
        )
        Z = V_p
        prev_p = p
        # Verify that the first layer equals the inputs
        if i == 0:
            assert V_p == mle_eval(matmul_layer(values), p)
        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])
    return True


def test():
    coord1 = koalabear.ExtendedKoalaBear([123, 45, 678, 90])
    coord2 = koalabear.ExtendedKoalaBear([2718, 2818, 284590, 45])
    weights = chi_weights(coord1)
    eval1 = mle_eval(weights, coord2)
    eval2 = chi_eval(coord1, coord2)
    assert eval1 == eval2
    print("Chi polynomial tests passed")
    values = koalabear.KoalaBear(list(range(8192)))
    t1 = time.time()
    outputs = permutation(values)
    print("Raw execution time:", time.time() - t1)
    t2 = time.time()
    Z_top, proof = gkr_prove(values)
    print("Proof generated")
    print("Generation time:", time.time() - t2)
    assert gkr_verify(values, outputs, Z_top, proof)
    print("Verification completed")

if __name__ == '__main__':
    test()
