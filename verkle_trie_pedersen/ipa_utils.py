import blst
import pippenger
from poly_utils import PrimeField
import hashlib
import time

#
# Utilities for dealing with polynomials in evaluation form
#
# A polynomial in evaluation for is defined by its values on DOMAIN,
# where DOMAIN is [omega**0, omega**1, omega**2, ..., omega**(WIDTH-1)]
# where omega is a WIDTH root of unity, i.e. omega**WIDTH % MODULUS == 1 
#
# Any polynomial of degree < WIDTH can be represented uniquely in this form,
# and many operations (such as multiplication and exact division) are more
# efficient.
#
# By precomputing the basis in Lagrange basis, we can also easily
# commit to a a polynomial in evaluation form.
#

def hash(x):
    if isinstance(x, bytes):
        return hashlib.sha256(x).digest()
    elif isinstance(x, blst.P1):
        return hash(x.compress())
    b = b""
    for a in x:
        if isinstance(a, bytes):
            b += a
        elif isinstance(a, int):
            b += a.to_bytes(32, "little")
        elif isinstance(a, blst.P1):
            b += hash(a.compress())
    return hash(b)




class IPAUtils():

    """
    Class that defines helper function for IPA proofs in evaluation form (Lagrange basis)
    """
    def __init__(self, BASIS_G, BASIS_Q, primefield):
        self.MODULUS = primefield.MODULUS
        self.BASIS_G = BASIS_G
        self.BASIS_Q = BASIS_Q
        self.WIDTH = primefield.WIDTH
        self.DOMAIN = primefield.DOMAIN
        self.primefield = primefield

        #self.BASIS_LAGRANGE = []
        #for i in range(primefield.WIDTH):
        #    g = blst.G1().mult(0)
        #    for b, e in zip(BASIS_G, primefield.lagrange_polys[i]):
        #        g.add(b.dup().mult(e))

        #    self.BASIS_LAGRANGE.append(g)


    def hash_to_field(self, x):
        return int.from_bytes(hash(x), "little") % self.MODULUS

    def pedersen_commit(self, a):
        """
        Returns a Pedersen commitment to the vector a (defined by its coefficients)
        """
        return pippenger.pippenger_simple(self.BASIS_G, a)

    def pedersen_commit_basis(self, a, basis):
        """
        Returns a Pedersen commitment to the vector a (defined by its coefficients)
        """
        return pippenger.pippenger_simple(basis, a)

    
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
        q = self.BASIS_Q.dup().mult(w)

        current_commitment = C.dup().add(q.dup().mult(y))
        current_basis = self.BASIS_G




        i = 0
        xs = []
        xinvs = []

        while n > 1:
            C_L, C_R = [blst.P1(C) for C in proof[i]]

            x = self.hash_to_field([C_L, C_R])
            xinv = self.primefield.inv(x)
            xs.append(x)
            xinvs.append(xinv)

            current_commitment = current_commitment.dup().add(C_L.dup().mult(x)).add(C_R.dup().mult(xinv))

            n = m
            m = n // 2
            i = i + 1

        f_g_coefs = self.f_g_coefs(xinvs)
        g_l = pippenger.pippenger_simple(self.BASIS_G, f_g_coefs)

        b_l = self.inner_product(b, f_g_coefs)

        a_l = proof[-1][0]

        a_l_times_b_l = a_l * b_l % self.MODULUS

        computed_commitment = g_l.mult(a_l).add(q.mult(a_l_times_b_l))

        return current_commitment.is_equal(computed_commitment)

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
        q = self.BASIS_Q.dup().mult(w)

        current_basis = self.BASIS_G

        time_a = time.time()

        while n > 1:
            # Reduction step

            a_L = a[:m]
            a_R = a[m:]
            b_L = b[:m]
            b_R = b[m:]
            z_L = self.inner_product(a_R, b_L)
            z_R = self.inner_product(a_L, b_R)
            C_L = self.pedersen_commit_basis(a_R, current_basis[:m]).add(q.dup().mult(z_L))
            C_R = self.pedersen_commit_basis(a_L, current_basis[m:]).add(q.dup().mult(z_R))

            proof.append([C_L.compress(), C_R.compress()])

            x = self.hash_to_field([C_L, C_R])
            xinv = self.primefield.inv(x)

            # Compute updates for next round
            a = [(v + x * w) % self.MODULUS for v, w in zip(a_L, a_R)]
            b = [(v + xinv * w) % self.MODULUS for v, w in zip(b_L, b_R)]

            current_basis = [v.dup().add(w.dup().mult(xinv)) for v, w in zip(current_basis[:m], current_basis[m:])]
            n = m
            m = n // 2


        time_b = time.time()
        print((time_b - time_a)*1000)

        # Final step
        proof.append([a[0]])

        return y, proof

if __name__ == "__main__":
    MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    WIDTH = 256

    time_a = time.time()
    BASIS_G = [blst.P1().hash_to(i.to_bytes(32, "little")) for i in range(WIDTH)]
    BASIS_Q = blst.P1().hash_to((256).to_bytes(32, "little"))
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
