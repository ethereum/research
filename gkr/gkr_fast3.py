from zorch import koalabear
from sumcheck import sumcheck_prove, sumcheck_verify
import time
from gkr_utils import (
    M, matmul_layer, chi_weights, mle_eval, chi_eval, compute_weights,
    fast_point_eval, generate_weights_seed_coords, generate_weights, log2, hash as _hash
)

def is_inner_round(i):
    return 4 <= i < 60

def permutation(values):
    for i in range(64):
        if is_inner_round(i):
            values = matmul_layer(values)
            values[..., ::16] **= 3
        else:
            values = matmul_layer(values)**3 + i
    return values

def hash(*inputs):
    return _hash(*inputs, permutation=permutation)

def first_col(vec):
    size = vec.shape[0] // 16
    return vec.reshape((size, 16))[:, 0]

def reverse_matmul_layer(values):  # M^T
    size = values.shape[0] // 16
    return koalabear.matmul(values.reshape((size, 16)), M.swapaxes(0,1)).reshape((size*16,))

def prepend_zeros4(coords):
    z = koalabear.ExtendedKoalaBear([0,0,0,0])
    return [z,z,z,z] + list(coords)

def fast_point_eval_with_skips(source_coords, eval_coords, skips=0):
    """
    Evaluate < (M^T)^(1+skips) * chi(source_coords[:4]) , chi(eval_coords[:4]) >
    times the row-part chi(source_coords[4:], eval_coords[4:]).
    - skips=0 reproduces your current fast_point_eval.
    - skips=1 accounts for one linear-only layer (e.g., a partial round) between reseeds.
    """
    # Start from chi on the 4 "column bits"
    vec = chi_weights(source_coords[:4])          # shape (16,)
    # Apply M^T a total of (1 + skips) times
    for _ in range(1 + skips):
        vec = koalabear.matmul(M.swapaxes(0,1), vec)
    # Evaluate against the eval_coords[:4]
    o = mle_eval(vec, eval_coords[:4])
    # Multiply by the row-part overlap
    o *= chi_eval(source_coords[4:], eval_coords[4:])
    return o

def gkr_prove(values):
    layers = [values]
    for i in range(64):
        if is_inner_round(i):
            new_layer = matmul_layer(layers[-1])
            new_layer[::16] **= 3
            layers.append(new_layer)
        else:
            layers.append(matmul_layer(layers[-1]) ** 3 + i)
    randomness = hash(layers[-1], values)
    weights = generate_weights(randomness, values.shape[0], hash)
    proof = []
    Z_top = koalabear.ExtendedKoalaBear.sum(layers[-1] * weights, axis=0)
    Z = Z_top
    for i in range(63, -1, -1):
        # Going in, we have an "obligation" to prove that
        # weights * (V**3 + i) sums to Z, initially Z_top
        V = matmul_layer(layers[i])
        if not is_inner_round(i):
            c0, c1, c2, c3, c4, p, W_p, V_p = sumcheck_prove(weights, V, 0, i, randomness, hash)
            # Now, we have an obligation to prove V(p) = V_p
            proof.append((c0, c1, c2, c3, c4, p, V_p))
            Z = V_p
            # V(p) equals the linear combination sum(chi_weights(p) * V). So we
            # compute the coeffs of that linear combination, and we now again
            # have an obligation in the format of the previous layer
            weights = compute_weights(p)
        else:
            W0 = first_col(weights)
            V0 = first_col(V)
            c0, c1, c2, c3, c4, p, W0_p, V0_p = sumcheck_prove(W0, V0, -1, 0, randomness, hash)
            proof.append((c0, c1, c2, c3, c4, p, V0_p))
            weights = reverse_matmul_layer(weights)
        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])

    return Z_top, proof

def gkr_verify(values, outputs, Z_top, proof):
    randomness = hash(outputs, values)
    weight_seed_coords = generate_weights_seed_coords(randomness, values.shape[0], hash)
    assert Z_top == mle_eval(outputs, weight_seed_coords)
    Z = Z_top
    prev_p = None
    skips = 0
    for i, (c0, c1, c2, c3, c4, p, V_p) in zip(range(63, -1, -1), proof):
        # The first auxiliary to the obligation reduction is the sumcheck
        if not is_inner_round(i):
            W_p = chi_eval(weight_seed_coords, p) if i==63 else fast_point_eval_with_skips(prev_p, p, skips)
            sumcheck_verify(c0, c1, c2, c3, c4, Z, p, W_p * (V_p ** 3 + i), randomness, hash)
            Z = V_p
            skips = 0
            prev_p = p
        else:
            W_p = fast_point_eval_with_skips(prev_p, prepend_zeros4(p), skips)
            partial_sum = c0[0] * 2 + c1[0] + c2[0] + c3[0] + c4[0]
            sumcheck_verify(c0, c1, c2, c3, c4, partial_sum, p, W_p * (V_p ** 3 - V_p), randomness, hash)
            Z -= partial_sum
            skips += 1

        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])
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
    values = koalabear.KoalaBear(list(range(524288)))
    t1 = time.time()
    outputs = permutation(values)
    print("Raw execution time:", time.time() - t1)
    t2 = time.time()
    Z_top, proof = gkr_prove(values)
    print("Proof generated")
    print("Generation time:", time.time() - t2)
    #assert gkr_verify(values, outputs, Z_top, proof)
    #print("Verification completed")

if __name__ == '__main__':
    test()
