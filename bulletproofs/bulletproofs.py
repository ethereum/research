from py_ecc import optimized_bls12_381 as b
from hashlib import sha256
from dataclasses import dataclass
from multicombs import lincomb
import time

# See page 25 and 29 of https://eprint.iacr.org/2020/1536.pdf and
# page 49-50 of https://eprint.iacr.org/2020/499.pdf

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
        if b.is_on_curve((x, y, b.FQ(1)), b.b):
            points.append(b.multiply((x, y, b.FQ(1)), BLS12_381_COFACTOR))
        x += b.FQ(1)
    return points


# Commit to some polynomial
def commit(generator_points, poly):
    # Equivalent (but faster) to this:
    # reduce(
    #    b.add,
    #    [b.multiply(pt, cf) for pt, cf in zip(generator_points[:len(poly)], poly)],
    #    b.Z1
    # )
    return lincomb(generator_points[:len(poly)], poly, b.add, b.Z1)


# Returne True iff x is a power of two
def is_power_of_two(x):
    return x and (x & (x-1) == 0)


# Serializes an elliptic curve point. Used for Fiat-Shamir.
def serialize_point(pt):
    pt = b.normalize(pt)
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
    # Crop the base points to just what we need
    points = points[:len(poly)]
    # Left-side points for the proof
    L = []
    # Right-side points for the proof
    R = []
    # Fiat-shamir randomness value
    r = hash(serialize_point(commitment))
    # log(n) rounds...
    while len(poly) > 1:
        # Generate the left-side and right-side points
        polyL, polyR = left_half(poly), right_half(poly)
        pointsL, pointsR = left_half(points), right_half(points)
        yL = commit(pointsR, polyL)
        yR = commit(pointsL, polyR)
        L.append(yL)
        R.append(yR)
        # Generate random coefficient for recombining the L and R and commitment
        r = hash(r + serialize_point(yL) + serialize_point(yR))
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        # Generate half-size polynomial and points for the next round
        poly = [(cL + cR * a) % b.curve_order for (cL, cR) in zip(polyL, polyR)]
        points = [b.add(b.multiply(pL, a), pR) for (pL, pR) in zip(pointsL, pointsR)]
        # print('intermediate commitment:', commit(points, poly))
    return Proof(L, R, poly[0])


def verify(points, commitment, proof):
    # Crop the base points to just what we need
    points = points[:2**len(proof.L)]
    # Fiat-shamir randomness value
    r = hash(serialize_point(commitment))
    # For verification, we need to generate the same random linear combination of
    # base points that the prover did.. But because we don't need to use it until
    # the end, we do it more efficiently here: when we progress through the rounds,
    # we keep track of how many times each points[i] will appear in the final
    # result...
    points_coeffs = [1]
    # log(n) rounds, just like the prover...
    for i in range(len(proof.L)):
        r = hash(r + serialize_point(proof.L[i]) + serialize_point(proof.R[i]))
        # Generate random coefficient for recombining (same as the prover)
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        # Add L and R into the commitment, applying the appropriate coefficients
        commitment = b.add(
            proof.L[i],
            b.add(
                b.multiply(commitment, a),
                b.multiply(proof.R[i], a**2)
            )
        )
        # print('intermediate commitment:', commitment)
        # Update the coefficients (points_coeffs[i] = how many times points[i] will
        # appear in the single base point of the last round)
        points_coeffs = sum([[(x*a) % b.curve_order, x] for x in points_coeffs], [])
    # Finally, we do the linear combination
    combined_point = lincomb(points, points_coeffs, b.add, b.Z1)
    # Base case check: base_point * coefficient ?= commitment
    return b.eq(b.multiply(combined_point, proof.tip), commitment)


# Prove that `commitment` actually is the commitment to a polynomial
# `p` such that `p(x) = y`
def prove_evaluation(points, commitment, poly, x, y):
    assert is_power_of_two(len(poly))
    # Crop the base points to just what we need. We add an additional base point,
    # which we will use to mix in the _evaluation_ of the polynomial.
    points, H = points[:len(poly)], points[len(poly)]
    # Alongside the base points, we track the powers of the x coordinate we are
    # proving an evaluation for. These points get manipulated in the same way as the
    # base points do.
    xpowers = [pow(x, i, b.curve_order) for i in range(len(poly))]
    # Left-side points for the proof
    L = []
    # Right-side points for the proof
    R = []
    # Fiat-shamir randomness value
    r = hash(serialize_point(commitment) + x.to_bytes(32, 'little') + y.to_bytes(32, 'little'))
    # For security, we randomize H
    H = b.multiply(H, int.from_bytes(r, 'little') % b.curve_order)
    while len(poly) > 1:
        # Generate the left-side and right-side points, except we also mix in a similarly
        # constructed "commitment" that uses `H * powers of x` as its base instead of the
        # base points.
        polyL, polyR = left_half(poly), right_half(poly)
        pointsL, pointsR = left_half(points), right_half(points)
        xpowersL, xpowersR = left_half(xpowers), right_half(xpowers)
        yL = commit(pointsR, polyL)
        yR = commit(pointsL, polyR)
        L.append(b.add(yL, b.multiply(H, sum(a*b for a,b in zip(xpowersR, polyL)))))
        R.append(b.add(yR, b.multiply(H, sum(a*b for a,b in zip(xpowersL, polyR)))))
        # Generate random coefficient for recombining the L and R and commitment
        r = hash(r + serialize_point(L[-1]) + serialize_point(R[-1]))
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        # Generate half-size polynomial and points for the next round. Notice how we treat
        # the powers of x the same way that we do the base points
        poly = [(cL + cR * a) % b.curve_order for (cL, cR) in zip(polyL, polyR)]
        points = [b.add(b.multiply(pL, a), pR) for (pL, pR) in zip(pointsL, pointsR)]
        xpowers = [(xL * a + xR) % b.curve_order for (xL, xR) in zip(xpowersL, xpowersR)]
        # print('intermediate commitment:', b.add(commit(points, poly), b.multiply(H, sum(a*b for a,b in zip(xpowers, poly)))))
    return Proof(L, R, poly[0])


# Verify a proof of an evaluation made using the above protocol
def verify_evaluation(points, commitment, proof, x, y):
    # Crop the base points to just what we need. We add an additional base point,
    # which we will use to mix in the _evaluation_ of the polynomial.
    points, H = points[:2**len(proof.L)], points[2**len(proof.L)]
    # Powers of x, as in the prover
    xpowers = [pow(x, i, b.curve_order) for i in range(len(poly))]
    # Fiat-shamir randomness value
    r = hash(serialize_point(commitment) + x.to_bytes(32, 'little') + y.to_bytes(32, 'little'))
    # For security, we randomize H
    H = b.multiply(H, int.from_bytes(r, 'little') % b.curve_order)
    # We "mix in" H * the claimed evaluation P(x) = y. Notice that `H * P(x)` equals the
    # dot-product of `H * powers of x` and the polynomial coefficients, so it has the
    # "same format" as the polynomial commitment itself. This allows us to verify the
    # evaluation using the same technique that we use to just prove that the commitment
    # is valid
    commitment = b.add(commitment, b.multiply(H, y))
    # Track the linear combination so we can generate the final-round point and xpower,
    # just as before
    points_coeffs = [1]
    for i in range(len(proof.L)):
        # Generate random coefficient for recombining (same as the prover)
        r = hash(r + serialize_point(proof.L[i]) + serialize_point(proof.R[i]))
        a = int.from_bytes(r, 'little') % b.curve_order
        # print('a value: ', a)
        # Add L and R into the commitment, applying the appropriate coefficients
        commitment = b.add(
            proof.L[i],
            b.add(
                b.multiply(commitment, a),
                b.multiply(proof.R[i], a**2)
            )
        )
        # print('intermediate commitment:', commitment)
        # Update the coefficients (as in basic verification above)
        points_coeffs = sum([[(x*a) % b.curve_order, x] for x in points_coeffs], [])
    # Finally, we do the linear combination; same one for base points and x powers
    combined_point = lincomb(points, points_coeffs, b.add, b.Z1)
    combined_x_powers = sum(p*c for p,c in zip(xpowers, points_coeffs))
    # Base case check: base_point * coefficient ?= commitment. Note that here we
    # have to also mix H * the combined xpower into the final base point
    return b.eq(
        b.add(
            b.multiply(combined_point, proof.tip),
            b.multiply(H, (proof.tip * combined_x_powers) % b.curve_order)
        ),
        commitment
    )


time_cache = [time.time()]


def get_time_delta():
    time_cache.append(time.time())
    return time_cache[-1] - time_cache[-2]

if __name__ == '__main__':
    get_time_delta()
    points = mk_generator_points(32)
    print("Generated generator points: {:.3f}s".format(get_time_delta()))
    poly = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
    commitment = commit(points, poly)
    print("Simple commitment generated: {:.3f}s".format(get_time_delta()))
    proof = prove(points, commitment, poly)
    print("Proof generated: {:.3f}s".format(get_time_delta()))
    print(proof)
    assert verify(points, commitment, proof)
    print("Proof verified: {:.3f}s".format(get_time_delta()))
    proof2 = prove_evaluation(points, commitment, poly, 10, 3979853562951413)
    print("Evaluation proof generated: {:.3f}s".format(get_time_delta()))
    assert verify_evaluation(points, commitment, proof2, 10, 3979853562951413)
    print("Evaluation proof verified: {:.3f}s".format(get_time_delta()))
