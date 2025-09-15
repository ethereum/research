import numpy as np
from zorch import koalabear

def deg4_lagrange_weights(x):
    """
    Return weights w_{-2}, w_{-1}, w_0, w_1, w_2 such that
    for any degree-4 polynomial p,
      p(x) = sum_k w_k * p(k)  with k in {-2,-1,0,1,2}.
    Output order is (-2, -1, 0, 1, 2).
    """
    nodes = (-2, -1, 0, 1, 2)
    denoms = (24, -6, 4, -6, 24)  # Π_{m≠k} (k - m) for k = -2,-1,0,1,2
    coeffs = []
    for idx, k in enumerate(nodes):
        num = koalabear.KoalaBear(1)
        for m in nodes:
            if m != k:
                num *= (x - m)
        coeffs.append(num / denoms[idx])
    return tuple(coeffs)

# Proves sumcheck for W * (V**3 + coeff1 * V + coeff0)
def sumcheck_prove(W, V, coeff1, coeff0, randomness, hash):
    assert (W.shape[0] & (W.shape[0]-1)) == 0, "length must be power of two"
    """
    We start with a cube of evaluations of W and V at 2^k positions {0,1}^k,
    and we want to prove the sum of evaluations of
    W * (V**3 + coeff1 * V + coeff0) at all coordinates in the cube.
    
    At each step in the sumcheck, we commit to the deg-4 1D polynomial P
    where P(X) is the sum of all evaluations at (X, {0,1}, {0,1} ... {0,1}).
    We choose a random coordinate x, and from that point forward continue with
    the half-size hypercube at (x, {0,1}^(k-1)). We keep proceeding like this
    until we get to a single evaluation at some (x1, x2 ... xk).

    We have not _proven_ that evaluation, rather we've _reduced_ the claim
    about a sum of evaluations to a claim about a single evaluation at a point.
    """
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    size = W.shape[0]
    coords = []
    while size > 1:
        # We commit to the 1D polynomial via its evaluations at -2, -1, 0, 1, 2
        # We compute these evaluations, first for W and V
        W0 = W[::2]
        W1 = W[1::2]
        Wm = W1 - W0
        W2 = W1 + Wm
        Wm1 = W0 - Wm
        Wm2 = Wm1 - Wm
        V0 = V[::2]
        V1 = V[1::2]
        Vm = V1 - V0
        V2 = V1 + Vm
        Vm1 = V0 - Vm
        Vm2 = Vm1 - Vm
        # And then for W * (V**3 + coeff1 * V + coeff0)
        _cls = W0.__class__
        c0.append(_cls.sum(Wm2 * (Vm2 ** 3 + coeff1 * Vm2 + coeff0), axis=0))
        c1.append(_cls.sum(Wm1 * (Vm1 ** 3 + coeff1 * Vm1 + coeff0), axis=0))
        c2.append(_cls.sum(W0 * (V0 ** 3 + coeff1 * V0 + coeff0), axis=0))
        c3.append(_cls.sum(W1 * (V1 ** 3 + coeff1 * V1 + coeff0), axis=0))
        c4.append(_cls.sum(W2 * (V2 ** 3 + coeff1 * V2 + coeff0), axis=0))
        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        W = W0 + coords[-1] * Wm # m = "slope" (W and V are both multilinear)
        V = V0 + coords[-1] * Vm
        size //= 2
    return c0, c1, c2, c3, c4, coords, W[0], V[0]

def sumcheck_verify(c0, c1, c2, c3, c4, total, coords, value, randomness, hash):
    for i, (v0, v1, v2, v3, v4, coord) in enumerate(zip(c0, c1, c2, c3, c4, coords)):
        randomness = hash(randomness, v0, v1, v2, v3, v4)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        # Check that the expected sum matches the provided polynomial
        if total is not None:
            assert total == v2 + v3
        # Compute the expected sum for the next round
        coeffs = deg4_lagrange_weights(coord)
        total = (
            v0 * coeffs[0] +
            v1 * coeffs[1] +
            v2 * coeffs[2] +
            v3 * coeffs[3] +
            v4 * coeffs[4]
        )
    assert total == value
    return True

def test():
    # dummy hash function
    def hash(*args):
        o = koalabear.KoalaBear.zeros((8,))
        p = 1
        for arg in args:
            buffer = koalabear.KoalaBear(arg.value.reshape((-1,)))
            for i in range(buffer.shape[0]):
                o[i%8] += buffer[i] * p
                p *= 37
        return koalabear.KoalaBear(o)

    W = koalabear.KoalaBear([3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 7950])
    V = koalabear.KoalaBear([2, 7,  18, 28, 18, 28, 45, 90, 45, 23, 53, 60,  2,  8,  7,    5])
    k = 63
    # V = koalabear.KoalaBear([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
    c0, c1, c2, c3, c4, coords, W_value, V_value = sumcheck_prove(W, V, 0, k, hash(W, V), hash)
    value = W_value * (V_value ** 3 + k)
    print("Proof generated")
    total = koalabear.KoalaBear.sum(W * (V**3 + k), axis=0)
    assert sumcheck_verify(c0, c1, c2, c3, c4, total, coords, value, hash(W, V), hash)
    print("Proof verified")

if __name__ == '__main__':
    test()
