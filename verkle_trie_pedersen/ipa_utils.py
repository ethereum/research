import blst
import pippenger
from poly_utils import PrimeField

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

class IPAUtils():

    """
    Class that defines helper function for IPA proofs in evaluation form (Lagrange basis)
    """
    def __init__(self, BASIS, primefield):
        self.MODULUS = primefield.MODULUS
        self.BASIS = BASIS
        self.WIDTH = primefield.WIDTH
        self.DOMAIN = primefield.DOMAIN
        self.primefield = primefield

        self.BASIS_LAGRANGE = []
        for i in range(primefield.WIDTH):
            g = blst.G1().mult(0)
            for b, e in zip(BASIS, primefield.lagrange_polys[i]):
                g.add(b.dup().mult(e))

            self.BASIS_LAGRANGE.append(g)


    def pedersen_commit_coef(self, f):
        """
        Returns a Pedersen commitment to the function f (defined by its coefficients)
        """
        return pippenger.pippenger_simple(self.BASIS, f)

    def pedersen_commit_lagrange(self, f_eval):
        """
        Returns a Pedersen commitment to the function f, defined by its evaluation on DOMAIN
        """
        return pippenger.pippenger_simple(self.BASIS_LAGRANGE, f_eval)


    def check_ipa_proof(self, C, z, y, proof):
        """
        Check the KZG proof 
        e(C - [y], [1]) = e(pi, [s - z])
        which is equivalent to
        e(C - [y], [1]) * e(-pi, [s - z]) == 1
        """
        pairing = blst.PT(blst.G2().to_affine(), C.dup().add(blst.G1().mult(y).neg()).to_affine())
        pairing.mul(blst.PT(self.SETUP["g2"][1].dup().add(blst.G2().mult(z).neg()).to_affine(), pi.dup().neg().to_affine()))

        return pairing.final_exp().is_one()


    def evaluate_and_compute_ipa_proof(self, f, z):
        """
        Evaluates a function f (given in evaluation form) at a point z (which can be in the DOMAIN or not)
        and gives y = f(z) as well as an IPA proof that this is the correct result
        """

        assert len(f) == len(self.DOMAIN)

        if z in self.DOMAIN:
            index = self.DOMAIN.index(z)
            y = f[index]
        else:
            y = self.evaluate_polynomial_in_evaluation_form(f, z)
        
        n = len(self.DOMAIN)
        m = n // 2

        a = []

        proof = []

        while n > 1:

        

        return y, pippenger.pippenger_simple(self.SETUP["g1_lagrange"], q)


    def compute_commitment_lagrange(self, values):
        """
        Computes a commitment for a function given in evaluation form.
        'values' is a dictionary and can have missing indices, which improves efficiency.
        """
        commitment = pippenger.pippenger_simple([self.SETUP["g1_lagrange"][i] for i in values.keys()], values.values())
        return commitment

if __name__ == "__main__":
    MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    WIDTH = 4

    BASIS = [blst.P1().hash_to(i.to_bytes(32, "little")) for i in range(WIDTH)]

    primefield = PrimeField(MODULUS, WIDTH)
    ipautils = IPAUtils(BASIS, primefield)

    poly = [3, 4, 3, 2]
    poly_eval = [primefield.eval_poly_at(poly, x) for x in primefield.DOMAIN]

    commit = ipautils.pedersen_commit_coef(poly)
    commit_eval = ipautils.pedersen_commit_lagrange(poly_eval)

    assert commit.is_equal(commit_eval)