# THIS IS EXPERIMENTAL CODE. DO NOT USE IN PRODUCTION!

from hashlib import sha256
hash_to_int = lambda x: int.from_bytes(sha256(x).digest(), 'little')

from py_ecc import bn128 as curve
POINT = tuple
SIG2 = tuple
SIG3 = tuple

def serialize_int(x):
    return x.to_bytes(32, 'little')

def serialize_point(pt):
    x, y = pt
    return x.n.to_bytes(32, 'little') + y.n.to_bytes(32, 'little')

def privtopub(key):
    return curve.multiply(curve.G1, key)

# Linear combination of a list of points and values
def lincomb(pts: list, values: list) -> POINT:
    o = curve.Z1
    for pt, value in zip(pts, values):
        o = curve.add(o, curve.multiply(pt, value))
    return o

# Make a Schnorr signature
def sign(msg: bytes, key: int) -> SIG2:
    r = hash_to_int(msg + serialize_int(key))
    R = curve.multiply(curve.G1, r)
    e = hash_to_int(serialize_point(R) + msg)
    s = (r - key * e) % curve.curve_order
    return (s, e)

# Verify a Schnorr signature
def verify(msg: bytes, KEY: POINT, sig: SIG2) -> bool:
    s, e = sig
    # Gs + Ke = G(r - ke) + Ke = Gr - Ke + Ke = R
    R = curve.add(curve.multiply(curve.G1, s), curve.multiply(KEY, e))
    new_e = hash_to_int(serialize_point(R) + msg)
    return new_e == e

# Make a 1-of-2 signature, knowing the first of 2 keys
def sign_firstof2(msg: bytes, key1: int, KEY2: POINT, BASE: POINT =curve.G1) -> SIG3:
    r1 = hash_to_int(msg + serialize_int(key1))
    R1 = curve.multiply(BASE, r1)
    e1 = hash_to_int(serialize_point(R1) + msg)
    s2 = hash_to_int(msg + serialize_int(key1) + b'\x01') % curve.curve_order
    R2 = curve.add(curve.multiply(BASE, s2), curve.multiply(KEY2, e1))
    new_e = hash_to_int(serialize_point(R2) + msg)
    s1 = (r1 - key1 * new_e) % curve.curve_order
    return (s1, s2, new_e)

# Make a 1-of-2 signature, knowing the second of 2 keys
def sign_secondof2(msg: bytes, KEY1: POINT, key2: int, BASE: POINT =curve.G1) -> SIG3:
    r2 = hash_to_int(msg + serialize_int(key2))
    R2 = curve.multiply(BASE, r2)
    e2 = hash_to_int(serialize_point(R2) + msg)
    s1 = hash_to_int(msg + serialize_int(key2) + b'\x01') % curve.curve_order
    R1 = curve.add(curve.multiply(BASE, s1), curve.multiply(KEY1, e2))
    new_e = hash_to_int(serialize_point(R1) + msg)
    s2 = (r2 - key2 * new_e) % curve.curve_order
    return (s1, s2, e2)
 
# Verify a 1-of-2 signature
def verify_1of2(msg: bytes, KEY1: POINT, KEY2: POINT, sig: SIG3, BASE: POINT =curve.G1) -> bool:
    s1, s2, e = sig
    R1 = curve.add(curve.multiply(BASE, s1), curve.multiply(KEY1, e))
    new_e = hash_to_int(serialize_point(R1) + msg)
    R2 = curve.add(curve.multiply(BASE, s2), curve.multiply(KEY2, new_e))
    newer_e = hash_to_int(serialize_point(R2) + msg)
    return newer_e == e

# Generate points C1, D1, C2, D2, which equal
# EITHER (A1*f, B1*f, A2*f, B2*f) OR (A2*f, B2*f, A1*f, B1*f)
# And generate a proof that this was done correctly
def prove_blind_and_swap(A1: POINT, B1: POINT, A2: POINT, B2: POINT, factor: int, swap=False):
    # Compute the blind-and-swap
    if not swap:
        C1, D1, C2, D2 = (curve.multiply(P, factor) for P in (A1, B1, A2, B2))
    else:
        C1, D1, C2, D2 = (curve.multiply(P, factor) for P in (A2, B2, A1, B1))
    # Fiat shamir to choose a random linear combination
    msg = b''.join(serialize_point(x) for x in (A1, B1, A2, B2, C1, C2, D1, D2))
    r = hash_to_int(msg + b'\x01')
    # Take that linear combination of the base
    BASE = lincomb((A1, B1, A2, B2), (1, r, r**2, r**3))
    # The PUB_NOSWAP point is the same linear combination of (C1, D1, C2, D2)
    # The PUB_WITHSWAP point is the same linear combination of (C2, D2, C1, D1)
    # If you are not swapping, then PUB_NOSWAP = factor * BASE
    # If you are swapping, then PUB_WITHSWAP = factor * BASE
    # So we now transformed the problem into a 1-of-2 ringsig
    if not swap:
        PUB_WITHSWAP = lincomb((C2, D2, C1, D1), (1, r, r**2, r**3))
        proof = sign_firstof2(msg, factor, PUB_WITHSWAP, BASE)
    else:
        PUB_NOSWAP = lincomb((C1, D1, C2, D2), (1, r, r**2, r**3))
        proof = sign_secondof2(msg, PUB_NOSWAP, factor, BASE)
    return C1, D1, C2, D2, proof

# Verify a proof of a blind-and-swap operation
def verify_blind_and_swap(A1: POINT, B1: POINT, A2: POINT, B2: POINT,
                          C1: POINT, D1: POINT, C2: POINT, D2: POINT,
                          proof: SIG3):
    msg = b''.join(serialize_point(x) for x in (A1, B1, A2, B2, C1, C2, D1, D2))
    r = hash_to_int(msg + b'\x01')
    BASE = lincomb((A1, B1, A2, B2), (1, r, r**2, r**3))
    PUB_NOSWAP = lincomb((C1, D1, C2, D2), (1, r, r**2, r**3))
    PUB_WITHSWAP = lincomb((C2, D2, C1, D1), (1, r, r**2, r**3))
    return verify_1of2(msg, PUB_NOSWAP, PUB_WITHSWAP, proof, BASE)

def test():
    # Basic schnorr
    key1, key2 = 1337, 42069
    KEY1, KEY2 = privtopub(key1), privtopub(key2)
    sig = sign(b'cow', key1)
    assert verify(b'cow', KEY1, sig)
    print("Passed basic schnorr test")
    # 1-of-2 signatures
    BASE = curve.multiply(curve.G1, 1)
    firstof2_sig = sign_firstof2(b'cow', key1, KEY2, BASE)
    secondof2_sig = sign_secondof2(b'cow', KEY1, key2, BASE)
    assert verify_1of2(b'cow', KEY1, KEY2, firstof2_sig, BASE)
    assert verify_1of2(b'cow', KEY1, KEY2, secondof2_sig, BASE)
    print("Passed 1 of 2 signature test")
    # Blind and swap proofs
    A1, B1, A2, B2 = (curve.multiply(curve.G1, x) for x in (31337, 69042, 8675309, 299792458))
    factor = 5
    C1, D1, C2, D2, proof = prove_blind_and_swap(A1, B1, A2, B2, factor, False)
    FAKE_POINT = curve.multiply(curve.G1, 98765432123456789)
    assert (C1, D1, C2, D2) == tuple(curve.multiply(P, factor) for P in (A1, B1, A2, B2))
    assert verify_blind_and_swap(A1, B1, A2, B2, C1, D1, C2, D2, proof)
    assert not verify_blind_and_swap(A1, B1, A2, B2, C1, FAKE_POINT, C2, D2, proof)
    factor2 = 7
    E1, F1, E2, F2, proof = prove_blind_and_swap(C1, D1, C2, D2, factor2, True)
    assert (E1, F1, E2, F2) == tuple(curve.multiply(P, factor2) for P in (C2, D2, C1, D1))
    assert verify_blind_and_swap(C1, D1, C2, D2, E1, F1, E2, F2, proof)
    assert not verify_blind_and_swap(C1, D1, C2, D2, E1, F1, E2, FAKE_POINT, proof)
    print("Passed blind-and-swap test")

if __name__ == '__main__':
    test()
