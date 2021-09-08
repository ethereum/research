from bandersnatch import Point, Scalar
from poly_utils import PrimeField
import hashlib
import time

#
# Utilities for dealing with polynomials in evaluation form
#
# A polynomial in evaluation for is defined by its values on DOMAIN,
# where DOMAIN is [0, 1, 2, ...]
#
# Any polynomial of degree < WIDTH can be represented uniquely in this form,
# and many operations (such as multiplication and exact division) are more
# efficient.
#
#

def hash(x):
    if isinstance(x, bytes):
        return hashlib.sha256(x).digest()
    elif isinstance(x, Point):
        return hash(x.serialize())
    b = b""
    for a in x:
        if isinstance(a, bytes):
            b += a
        elif isinstance(a, int):
            b += a.to_bytes(32, "little")
        elif isinstance(a, Point):
            b += hash(a.serialize())
    return hash(b)

class IPAUtils():
    """
    Class that defines helper functions for IPA proofs in evaluation form (Lagrange basis)
    """

    def __init__(self, BASIS_G, BASIS_Q, primefield):
        self.MODULUS = primefield.MODULUS
        self.BASIS_G = BASIS_G
        self.BASIS_Q = BASIS_Q
        self.WIDTH = primefield.WIDTH
        self.DOMAIN = primefield.DOMAIN
        self.primefield = primefield


    def hash_to_field(self, x):
        return int.from_bytes(hash(x), "little") % self.MODULUS


    def pedersen_commit(self, a):
        """
        Returns a Pedersen commitment to the vector a (defined by its coefficients)
        """
        return Point().msm(self.BASIS_G, [Scalar().from_int(x) for x in a])


    def pedersen_commit_sparse(self, values):
        """
        Returns a Pedersen commitment to the vector a (defined by its coefficients)
        """
        if len(values) < 5:
            if len(values) == 0:
                return Point().mul(0)
            else:
                it = iter(values.items())
                k, v = next(it)
                r = self.BASIS_G[k].dup().glv(v)
                for k, v in it:
                    r = r.add(self.BASIS_G[k].dup().glv(v))
                return r
        return Point().msm([self.BASIS_G[i] for i in values.keys()], [Scalar().from_int(x) for x in values.values()])


    def pedersen_commit_basis(self, a, basis):
        """
        Returns a Pedersen commitment to the vector a (defined by its coefficients)
        """
        return Point().msm(basis, [Scalar().from_int(x) for x in a])

    
    def f_g_coefs(self, xinv_vec):
        f_g_coefs = []

        for i in range(len(self.DOMAIN)):
            binary = [int(x) for x in bin(i)[2:].rjust(len(xinv_vec), '0')]
            coef = 1
            for xinv, b in zip(xinv_vec, binary):
                if b == 1:
                    coef = coef * xinv % self.MODULUS
            f_g_coefs.append(coef)

        return f_g_coefs


    def check_ipa_proof(self, C, z, y, proof):
        """
        Check the IPA proof for a commitment to a Polynomial in evaluation form
        """
        n = len(self.DOMAIN)
        m = n // 2

        b = self.primefield.barycentric_formula_constants(z)

        w = self.hash_to_field([C, z, y])
        q = self.BASIS_Q.dup().glv(w)

        current_commitment = C.dup().add(q.dup().glv(y))
        current_basis = self.BASIS_G

        i = 0
        xs = []
        xinvs = []

        while n > 1:
            C_L, C_R = [Point().deserialize(C) for C in proof[i]]

            x = self.hash_to_field([C_L, C_R])
            xinv = self.primefield.inv(x)
            xs.append(x)
            xinvs.append(xinv)

            current_commitment = current_commitment.dup().add(C_L.dup().glv(x)).add(C_R.dup().glv(xinv))

            n = m
            m = n // 2
            i = i + 1

        f_g_coefs = self.f_g_coefs(xinvs)
        g_l = Point().msm(self.BASIS_G, f_g_coefs)

        b_l = self.inner_product(b, f_g_coefs)

        a_l = proof[-1][0]

        a_l_times_b_l = a_l * b_l % self.MODULUS

        computed_commitment = g_l.glv(a_l).add(q.glv(a_l_times_b_l))

        return current_commitment == computed_commitment


    def inner_product(self, a, b):
        return sum(x * y % self.MODULUS for x, y in zip(a, b)) % self.MODULUS


    def evaluate_and_compute_ipa_proof(self, C, f_eval, z):
        """
        Evaluates a function f (given in evaluation form) at a point z (which can be in the DOMAIN or not)
        and gives y = f(z) as well as an IPA proof that this is the correct result
        """

        assert len(f_eval) == len(self.DOMAIN)

        n = len(self.DOMAIN)
        m = n // 2

        a = f_eval[:]
        b = self.primefield.barycentric_formula_constants(z)
        y = self.inner_product(a, b)

        proof = []

        w = self.hash_to_field([C, z, y])
        q = self.BASIS_Q.dup().glv(w)

        current_basis = self.BASIS_G

        while n > 1:
            # Reduction step

            a_L = a[:m]
            a_R = a[m:]
            b_L = b[:m]
            b_R = b[m:]
            z_L = self.inner_product(a_R, b_L)
            z_R = self.inner_product(a_L, b_R)
            C_L = self.pedersen_commit_basis(a_R, current_basis[:m]).add(q.dup().glv(z_L))
            C_R = self.pedersen_commit_basis(a_L, current_basis[m:]).add(q.dup().glv(z_R))

            proof.append([C_L.serialize(), C_R.serialize()])

            x = self.hash_to_field([C_L, C_R])
            xinv = self.primefield.inv(x)

            # Compute updates for next round
            a = [(v + x * w) % self.MODULUS for v, w in zip(a_L, a_R)]
            b = [(v + xinv * w) % self.MODULUS for v, w in zip(b_L, b_R)]

            current_basis = [v.dup().add(w.dup().glv(xinv)) for v, w in zip(current_basis[:m], current_basis[m:])]
            n = m
            m = n // 2

        # Final step
        proof.append([a[0]])

        return y, proof

if __name__ == "__main__":
    MODULUS = 13108968793781547619861935127046491459309155893440570251786403306729687672801
    WIDTH = 256

    time_a = time.time()
    BASIS_G = [Point(generator=False) for i in range(WIDTH)]
    BASIS_Q = Point(generator=False)
    time_b = time.time()

    print("Basis computed in {:.2f} ms".format((time_b - time_a)*1000))

    time_a = time.time()
    primefield = PrimeField(MODULUS, WIDTH)
    time_b = time.time()

    print("Lagrange precomputes in {:.2f} ms".format((time_b - time_a)*1000))

    ipautils = IPAUtils(BASIS_G, BASIS_Q, primefield)

    poly = [3, 4, 3, 2, 1, 3, 3, 3]*32
    poly_eval = [primefield.eval_poly_at(poly, x) for x in primefield.DOMAIN]

    time_a = time.time()
    C = ipautils.pedersen_commit(poly_eval)
    time_b = time.time()

    print("Pedersen commitment computed in {:.2f} ms".format((time_b - time_a)*1000))

    time_a = time.time()
    y, proof = ipautils.evaluate_and_compute_ipa_proof(C, poly_eval, 17)
    time_b = time.time()

    print("Evaluation and proof computed in {:.2f} ms".format((time_b - time_a)*1000))

    time_a = time.time()
    assert ipautils.check_ipa_proof(C, 17, y, proof)
    time_b = time.time()

    print("Proof verified in {:.2f} ms".format((time_b - time_a)*1000))
