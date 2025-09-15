from zorch import koalabear
from sumcheck import sumcheck_prove, sumcheck_verify
import time
from gkr_utils import (
    M, M_inner, matmul_layer, inner_matmul_layer, chi_weights, mle_eval,
    chi_eval, compute_weights, fast_point_eval, generate_weights_seed_coords,
    generate_weights, log2, hash as _hash
)

# Here, we do a significantly more complicated and optimized GKR for a more
# optimized version of Poseidon. Specifically, only rounds 0...3 and 60...63
# are "full", in the other rounds we only cube the first value
def is_inner_round(i):
    return 4 <= i < 60

def permutation(values):
    for i in range(64):
        if is_inner_round(i):
            values = inner_matmul_layer(values)
            values[..., ::16] **= 3
        else:
            values = matmul_layer(values)**3 + i
    return values

def hash(*inputs):
    return _hash(*inputs, permutation=permutation)

# Takes as input a (1D) GKR state, outputs the sub-state corresponding
# to the first wire. Used for sumchecking the first wire only
def first_col(vec):
    size = vec.shape[0] // 16
    return vec.reshape((size, 16))[:, 0]

# Self-explanatory
def prepend_zeros4(coords):
    z = koalabear.ExtendedKoalaBear([0,0,0,0])
    return [z,z,z,z] + list(coords)

# Helper: save powers of the inner matrix
MINNER_POWERS = []
_I16 = koalabear.KoalaBear([[
    1 if i==j else 0 for j in range(16)] for i in range(16)
])

MINNER_POWERS.append(_I16)  # k = 0
for k in range(1, 65):      # fill up to the 64th power
    MINNER_POWERS.append(koalabear.matmul(MINNER_POWERS[-1], M_inner))

# Reusable e0 to extract the first column quickly (base-field length-16 vector)
_E0 = koalabear.KoalaBear([1] + [0]*15)

def fast_point_eval_with_skips(source_coords, eval_coords, skips=0):
    """
    Evaluate < (M^T)^(1+skips) * chi(source_coords[:4]), chi(eval_coords[:4]) >
    times the row-part chi(source_coords[4:], eval_coords[4:]).
    - skips=0 reproduces fast_point_eval.
    - skips=1 accounts for one linear-only layer (ie. a partial round)
    """
    # Start from chi on the 4 "column bits"
    vec = chi_weights(source_coords[:4])          # shape (16,)
    # Apply the full-round matrix once
    vec = koalabear.matmul(M, vec)
    # Apply the partial-round matrix `skips` times
    if skips > 0:
        vec = koalabear.matmul(MINNER_POWERS[skips], vec)
    # Evaluate against the eval_coords[:4]
    o = mle_eval(vec, eval_coords[:4])
    # Multiply by the row-part overlap
    o *= chi_eval(source_coords[4:], eval_coords[4:])
    return o

def first_col_after_inner_skips(weights, skips: int):
    """
    Return the first column of (M_inner^T)^skips * weights, without computing the
    other 15 columns. `weights` is an extension-field vector of length 16*size.
    """
    assert 0 <= skips <= 64, "skips must be in [0, 64]"
    size = weights.shape[0] // 16
    W = weights.reshape((size, 16))  # extension-field [size x 16]

    if skips == 0:
        return W[:, 0]

    # Base-field first column of (M_inner^T)^skips:
    # c = MINNER_POWERS[skips] @ e0  (length-16 base vector)
    c = koalabear.matmul(MINNER_POWERS[skips], _E0)

    # Dot each row with c (ext×base multiplies + ext adds), yielding an
    # extension-field length-`size` vector (the first column only).
    out = W[:, 0] * c[0]
    for j in range(1, 16):
        out += W[:, j] * c[j]
    return out

# Prove evaluations via a GKR algorithm
def gkr_prove(values):
    layers = [values]
    post_matmul_layers = []
    for i in range(64):
        if is_inner_round(i):
            new_layer = inner_matmul_layer(layers[-1])
            post_matmul_layers.append(new_layer)
            new_layer = new_layer.copy()
            new_layer[::16] **= 3
            layers.append(new_layer)
        else:
            new_layer = matmul_layer(layers[-1])
            post_matmul_layers.append(new_layer)
            layers.append(new_layer ** 3 + i)
    randomness = hash(layers[-1], values)
    weights = generate_weights(randomness, values.shape[0], hash)
    proof = []
    Z_top = koalabear.ExtendedKoalaBear.sum(layers[-1] * weights, axis=0)
    Z = Z_top
    skips = 0
    last_weights_full = weights
    for i in range(63, -1, -1):
        # Going in, we have an "obligation" to prove that
        # weights * (V**3 + i) sums to Z, initially Z_top
        if not is_inner_round(i):
            V = post_matmul_layers[i]
            if skips > 0:
                weights = matmul_layer(last_weights_full, MINNER_POWERS[skips])
            c0, c1, c2, c3, c4, p, W_p, V_p = \
                    sumcheck_prove(weights, V, 0, i, randomness, hash)
            # Now, we have an obligation to prove V(p) = V_p
            proof.append((c0, c1, c2, c3, c4, p, V_p))
            Z = V_p
            # V(p) equals the linear combination sum(chi_weights(p) * V). So we
            # compute the coeffs of that linear combination, and we now again
            # have an obligation in the format of the previous layer
            weights = compute_weights(p)
            last_weights_full = weights
            skips = 0
        else:
            # During inner rounds, we do a sumcheck only for the first wire. We
            # do not process the linear-only operation for the other 15 wires
            # directly; rather, we represent it via mixing powers of M_inner into
            # the weights matrix. Though even that we don't do directly, rather we
            # compute the first column of this matrix to make the sumcheck and
            # otherwise we just track how high the powers go
            V = post_matmul_layers[i]
            W0 = first_col_after_inner_skips(last_weights_full, skips)
            V0 = first_col(V)
            c0, c1, c2, c3, c4, p, W0_p, V0_p = \
                    sumcheck_prove(W0, V0, -1, 0, randomness, hash)
            proof.append((c0, c1, c2, c3, c4, p, V0_p))
            skips += 1
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
    assert Z_top == mle_eval(outputs, weight_seed_coords)
    Z = Z_top
    prev_p = None
    skips = 0
    for i, (c0, c1, c2, c3, c4, p, V_p) in zip(range(63, -1, -1), proof):
        # Full round logic is the same as the more basic version, except
        # we add a skips parameter to handle the case of the first full
        # round after many partial rounds
        if not is_inner_round(i):
            if i == 63:
                W_p = chi_eval(weight_seed_coords, p)
            else:
                W_p = fast_point_eval_with_skips(prev_p, p, skips)
            sumcheck_verify(
                c0, c1, c2, c3, c4, Z, p, W_p * (V_p**3 + i), randomness, hash
            )
            Z = V_p
            skips = 0
            prev_p = p
        else:
            # We verify the partial sumcheck, except we don't require it to
            # exactly match the previous round's sum. Instead, we "discharge"
            # the total sum that the partial sumcheck proves from Z, and leave
            # the rest inside Z. The "remainder" finally gets handled in the
            # first next full layer
            W_p = fast_point_eval_with_skips(prev_p, prepend_zeros4(p), skips)
            partial_sum = c2[0] + c3[0]
            sumcheck_verify(
                c0, c1, c2, c3, c4, partial_sum, p,
                W_p * (V_p ** 3 - V_p), randomness, hash
            )
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
    assert gkr_verify(values, outputs, Z_top, proof)
    print("Verification completed")

if __name__ == '__main__':
    test()
