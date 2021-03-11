from py_ecc import bls12_381 as b
from hashlib import sha256
from dataclasses import dataclass

# See page 25 and 29 of https://eprint.iacr.org/2020/1536.pdf and
# page 49-50 of https://eprint.iacr.org/2020/499.pdf

# ----------------------------------------------------------------------- #
# THIS IS AN EDUCATIONAL IMPLEMENTATION ONLY. DO NOT USE IN PRODUCTION!!! #
# ----------------------------------------------------------------------- #


BLS12_381_COFACTOR = 76329603384216526031706109802092473003


def hash(x):
    return sha256(x).digest()


# Creates the generator points. This is a public procedure that can be repeated
# by anyone, so it is NOT a trusted setup
def mk_generator_points(count):
    points = []
    x = b.FQ(1)
    while len(points) < count:
        y = (x ** 3 + b.b) ** ((b.field_modulus + 1) // 4)
        if b.is_on_curve((x, y), b.b):
            points.append(b.multiply((x, y), BLS12_381_COFACTOR))
        x += b.FQ(1)
    return points


# Commit to some polynomial
def commit(generator_points, poly):
    o = b.Z1
    for point, coeff in zip(generator_points[:len(poly)], poly):
        o = b.add(o, b.multiply(point, coeff))
    return o


# Returne True iff x is a power of two
def is_power_of_two(x):
    return x and (x & (x-1) == 0)


# Serializes an elliptic curve point. Used for Fiat-Shamir.
def serialize_point(pt):
    return pt[0].n.to_bytes(64, 'little') + pt[1].n.to_bytes(64, 'little')


# Returns the (left|right) half of something
def left_half(x):
    return x[:len(x)//2]


def right_half(x):
    return x[len(x)//2:]


# The data structure for a proof
@dataclass
class Proof():
    L: list
    R: list
    tip: int


# Prove that `commitment` actually is the commitment to a polynomial
# (it does not prove _which_ polynomial)
def prove(points, commitment, poly):
    assert is_power_of_two(len(poly))
    points = points[:len(poly)]
    L = []
    R = []
    r = hash(serialize_point(commitment))
    while len(poly) > 1:
        polyL, polyR = left_half(poly), right_half(poly)
        pointsL, pointsR = left_half(points), right_half(points)
        yL = commit(pointsR, polyL)
        yR = commit(pointsL, polyR)
        L.append(yL)
        R.append(yR)
        r = hash(r + serialize_point(yL) + serialize_point(yR))
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        poly = [(cL + cR * a) % b.curve_order for (cL, cR) in zip(polyL, polyR)]
        points = [b.add(b.multiply(pL, a), pR) for (pL, pR) in zip(pointsL, pointsR)]
        # print('intermediate commitment:', commit(points, poly))
    return Proof(L, R, poly[0])


def verify(points, commitment, proof):
    points = points[:2**len(proof.L)]
    r = hash(serialize_point(commitment))
    for i in range(len(proof.L)):
        r = hash(r + serialize_point(proof.L[i]) + serialize_point(proof.R[i]))
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        commitment = b.add(
            proof.L[i],
            b.add(
                b.multiply(commitment, a),
                b.multiply(proof.R[i], a**2)
            )
        )
        # print('intermediate commitment:', commitment)
        points = [
            b.add(b.multiply(pL, a), pR) for (pL, pR) in
            zip(left_half(points), right_half(points))
        ]
    return b.multiply(points[0], proof.tip) == commitment


# Prove that `commitment` actually is the commitment to a polynomial
# `p` such that `p(x) = y`
def prove_evaluation(points, commitment, poly, x, y):
    assert is_power_of_two(len(poly))
    points, H = points[:len(poly)], points[len(poly)]
    xpowers = [pow(x, i, b.curve_order) for i in range(len(poly))]
    L = []
    R = []
    r = hash(serialize_point(commitment))
    H = b.multiply(H, int.from_bytes(r, 'little') % b.curve_order)
    while len(poly) > 1:
        polyL, polyR = left_half(poly), right_half(poly)
        pointsL, pointsR = left_half(points), right_half(points)
        xpowersL, xpowersR = left_half(xpowers), right_half(xpowers)
        yL = commit(pointsR, polyL)
        yR = commit(pointsL, polyR)
        L.append(b.add(yL, b.multiply(H, sum(a*b for a,b in zip(xpowersR, polyL)))))
        R.append(b.add(yR, b.multiply(H, sum(a*b for a,b in zip(xpowersL, polyR)))))
        r = hash(r + serialize_point(L[-1]) + serialize_point(R[-1]))
        a = int.from_bytes(r, 'little') % b.curve_order
        print('a value: ', a)
        poly = [(cL + cR * a) % b.curve_order for (cL, cR) in zip(polyL, polyR)]
        points = [b.add(b.multiply(pL, a), pR) for (pL, pR) in zip(pointsL, pointsR)]
        xpowers = [(xL * a + xR) % b.curve_order for (xL, xR) in zip(xpowersL, xpowersR)]
        print('intermediate commitment:', b.add(commit(points, poly), b.multiply(H, sum(a*b for a,b in zip(xpowers, poly)))))
    return Proof(L, R, poly[0])


def verify_evaluation(points, commitment, proof, x, y):
    points, H = points[:2**len(proof.L)], points[2**len(proof.L)]
    xpowers = [pow(x, i, b.curve_order) for i in range(len(poly))]
    r = hash(serialize_point(commitment))
    H = b.multiply(H, int.from_bytes(r, 'little') % b.curve_order)
    commitment = b.add(commitment, b.multiply(H, y))
    for i in range(len(proof.L)):
        r = hash(r + serialize_point(proof.L[i]) + serialize_point(proof.R[i]))
        a = int.from_bytes(r, 'little') % b.curve_order
        print('a value: ', a)
        commitment = b.add(
            proof.L[i],
            b.add(
                b.multiply(commitment, a),
                b.multiply(proof.R[i], a**2)
            )
        )
        print('intermediate commitment:', commitment)
        points = [
            b.add(b.multiply(pL, a), pR) for (pL, pR) in
            zip(left_half(points), right_half(points))
        ]
        xpowers = [
            (xL * a + xR) % b.curve_order for (xL, xR) in
            zip(left_half(xpowers), right_half(xpowers))
        ]
    return b.add(
        b.multiply(points[0], proof.tip),
        b.multiply(H, (proof.tip * xpowers[0]) % b.curve_order)
    ) == commitment


if __name__ == '__main__':
    poly = [3, 1, 4, 1, 5, 9, 2, 6]
    points = mk_generator_points(32)
    commitment = commit(points, poly)
    proof = prove(points, commitment, poly)
    print(proof)
    assert verify(points, commitment, proof)
    print("Proof verified!")
    proof2 = prove_evaluation(points, commitment, poly, 10, 62951413)
    print("------------")
    assert verify_evaluation(points, commitment, proof2, 10, 62951413)
    print("Evaluation proof verified!")
