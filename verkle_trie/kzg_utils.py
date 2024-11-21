import blst
import pippenger

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
# By precomputing the trusted setup in Lagrange basis, we can also easily
# commit to a a polynomial in evaluation form.
#

class KzgUtils():

    """
    Class that defines helper function for Kate proofs in evaluation form (Lagrange basis)
    """
    def __init__(self, MODULUS, WIDTH, DOMAIN, SETUP, primefield):
        self.MODULUS = MODULUS
        self.WIDTH = WIDTH
        self.DOMAIN = DOMAIN
        self.SETUP = SETUP
        self.primefield = primefield
        # Precomputed inverses of 1 / (1 - DOMAIN[i])
        self.inverses = [0] + [primefield.inv(1 - DOMAIN[i]) for i in range(1, WIDTH)]
        self.inverse_width = primefield.inv(self.WIDTH)


    def evaluate_polynomial_in_evaluation_form(self, f, z):
        """
        Takes a polynomial in evaluation form and evaluates it at one point outside the domain. 
        Uses the barycentric formula:
        f(z) = (1 - z**WIDTH) / WIDTH  *  sum_(i=0)^WIDTH  (f(DOMAIN[i]) * DOMAIN[i]) / (z - DOMAIN[i])
        """
        r = 0
        for i in range(self.WIDTH):
            r += self.primefield.div(f[i] * self.DOMAIN[i], (z - self.DOMAIN[i]) )
        r = r * (pow(z, self.WIDTH, self.MODULUS) - 1) * self.inverse_width % self.MODULUS

        return r


    def compute_inner_quotient_in_evaluation_form(self, f, index):
        """
        Compute the quotient q(X) = (f(X) - f(DOMAIN[index])) / (X - DOMAIN[index]) in evaluation form.

        Inner means that the value z = DOMAIN[index] is one of the points at which f is evaluated -- so unlike an outer
        quotient (where z is not in DOMAIN), we need to do some extra work to compute q[index] where the formula above
        is 0 / 0
        """
        q = [0] * self.WIDTH
        y = f[index]
        for i in range(self.WIDTH):
            if i != index:
                q[i] = (f[i] - y) * self.DOMAIN[-i] * self.inverses[index - i] % self.MODULUS
                q[index] += - self.DOMAIN[(i - index) % self.WIDTH] * q[i] % self.MODULUS

        return q


    def compute_outer_quotient_in_evaluation_form(self, f, z, y):
        """
        Compute the quotient q(X) = (f(X) - y)) / (X - z) in evaluation form. Note that this only works if the quotient
        is exact, i.e. f(z) = y, and otherwise returns garbage
        """
        q = [0] * self.WIDTH
        for i in range(self.WIDTH):
            q[i] = self.primefield.div(f[i] - y, self.DOMAIN[i] - z)

        return q


    def check_kzg_proof(self, C, z, y, pi):
        """
        Check the KZG proof 
        e(C - [y], [1]) = e(pi, [s - z])
        which is equivalent to
        e(C - [y], [1]) * e(-pi, [s - z]) == 1
        """
        pairing = blst.PT(blst.G2().to_affine(), C.dup().add(blst.G1().mult(y).neg()).to_affine())
        pairing.mul(blst.PT(self.SETUP["g2"][1].dup().add(blst.G2().mult(z).neg()).to_affine(), pi.dup().neg().to_affine()))

        return pairing.final_exp().is_one()


    def evaluate_and_compute_kzg_proof(self, f, z):
        """
        Evaluates a function f (given in evaluation form) at a point z (which can be in the DOMAIN or not)
        and gives y = f(z) as well as a Kate proof that this is the correct result
        """
        if z in self.DOMAIN:
            index = self.DOMAIN.index(z)
            y = f[index]
            q = self.compute_inner_quotient_in_evaluation_form(f, index)
        else:
            y = self.evaluate_polynomial_in_evaluation_form(f, z)
            q = self.compute_outer_quotient_in_evaluation_form(f, z, y)

        return y, pippenger.pippenger_simple(self.SETUP["g1_lagrange"], q)


    def compute_commitment_lagrange(self, values):
        """
        Computes a commitment for a function given in evaluation form.
        'values' is a dictionary and can have missing indices, which improves efficiency.
        """
        commitment = pippenger.pippenger_simple([self.SETUP["g1_lagrange"][i] for i in values.keys()], values.values())
        return commitment