import numpy as np
from zorch import koalabear
from utils import hash

# Proves sumcheck for W * (V**3 + c)
def sumcheck_prove(W, V, constant, randomness):
    assert (W.shape[0] & (W.shape[0]-1)) == 0, "length must be power of two"
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    size = W.shape[0]
    coords = []
    while size > 1:
        Wb = W[::2]
        Wm = W[1::2] - Wb
        Vb = V[::2]
        Vm = V[1::2] - Vb
        _cls = Wb.__class__
        Vb_sq = Vb * Vb
        Vb_cb = Vb_sq * Vb
        Vm_sq = Vm * Vm
        Vm_cb = Vm_sq * Vm
        Vb_Vm_3 = Vb * Vm * 3
        c0.append(_cls.sum(Wb * (Vb_cb + constant), axis=0))
        c1.append(_cls.sum(Wm * (Vb_cb + constant) + Wb * Vb * Vb_Vm_3, axis=0))
        c2.append(_cls.sum(Wm * Vb * Vb_Vm_3 + Wb * Vm * Vb_Vm_3, axis=0))
        c3.append(_cls.sum(Wm * Vm * Vb_Vm_3 + Wb * Vm_cb, axis=0))
        c4.append(_cls.sum(Wm * Vm_cb, axis=0))
        randomness = hash(randomness, c0[-1], c1[-1], c2[-1], c3[-1], c4[-1])
        coords.append(koalabear.ExtendedKoalaBear(randomness[:4].value))
        W = Wb + coords[-1] * Wm
        V = Vb + coords[-1] * Vm
        size //= 2
    return c0, c1, c2, c3, c4, coords, W[0], V[0]

def sumcheck_verify(c0, c1, c2, c3, c4, total, coords, value, randomness):
    for i, (v0, v1, v2, v3, v4, coord) in enumerate(zip(c0, c1, c2, c3, c4, coords)):
        randomness = hash(randomness, v0, v1, v2, v3, v4)
        assert koalabear.ExtendedKoalaBear(randomness[:4].value) == coord
        if total is not None:
            assert total == v0 * 2 + v1 + v2 + v3 + v4
        total = v0 + v1 * coord + v2 * coord**2 + v3 * coord**3 + v4 * coord**4
    assert total == value
    return True

def test():
    W = koalabear.KoalaBear([3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26, 43, 38, 32, 7950])
    V = koalabear.KoalaBear([2, 7,  18, 28, 18, 28, 45, 90, 45, 23, 53, 60,  2,  8,  7,    5])
    k = 63
    # V = koalabear.KoalaBear([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])
    c0, c1, c2, c3, c4, coords, W_value, V_value = sumcheck_prove(W, V, k, hash(W, V))
    value = W_value * (V_value ** 3 + k)
    print("Proof generated")
    total = koalabear.KoalaBear.sum(W * (V**3 + k), axis=0)
    assert sumcheck_verify(c0, c1, c2, c3, c4, total, coords, value, hash(W, V))
    print("Proof verified")
