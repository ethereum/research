# Creates an object that includes convenience operations for numbers
# and polynomials in some prime field

# Also added interpolation over an arbitrary DOMAIN (not roots of unity)
#

class PrimeField():
    def __init__(self, MODULUS, WIDTH):
        assert pow(2, MODULUS, MODULUS) == 2
        self.WIDTH = WIDTH
        self.MODULUS = MODULUS
        self.DOMAIN = list(range(WIDTH))

        self.A = self.zpoly(self.DOMAIN)
        self.Aprime = self.formal_derivative(self.A)

        # i-th Lagrange polynomial
        self.lagrange_polys = []

        # Aprime evaluated on the DOMAIN
        self.Aprime_DOMAIN = []

        # Aprime on the DOMAIN, inverted
        self.Aprime_DOMAIN_inv = []
        for i, x in enumerate(self.DOMAIN):
            self.Aprime_DOMAIN.append(self.eval_poly_at(self.Aprime, x))
            self.Aprime_DOMAIN_inv.append(self.inv(self.Aprime_DOMAIN[-1]))
            self.lagrange_polys.append(self.mul_polys([self.Aprime_DOMAIN_inv[-1]], 
                        self.div_polys(self.A, [-x, 1])))

        # Inverses needed for quotients
        self.INVERSES = [self.inv(x) for x in list(range(WIDTH)) + list(range(-WIDTH + 1, 0))]

        
    def formal_derivative(self, f):
        return [(n + 1) * c % self.MODULUS for n, c in enumerate(f[1:])]


    def evaluate_polynomial_in_evaluation_form(self, f, z):
        """
        Takes a polynomial in evaluation form and evaluates it at one point outside the DOMAIN. 
        Uses the barycentric formula:
        f(z) = A(z) *  sum_(i=0)^(WIDTH-1)  f(DOMAIN[i]) / A'(DOMAIN[i]) * 1 / (z - DOMAIN[i])
        """
        r = 0
        Az = self.eval_poly_at(self.A, z)

        inverses = self.multi_inv([z - x for x in self.DOMAIN])

        for i, x in enumerate(inverses):
            r += f[i] * self.Aprime_DOMAIN_inv[i] * x % self.MODULUS

        
        r = r * Az % self.MODULUS

        return r


    def barycentric_formula_constants(self, z):
        """
        Gives the constant used in the barycentric formula when evaluating a polynomial at z
        b_i = A(z) / A'(DOMAIN[i]) * 1 / (z - DOMAIN[i])
        """
        r = []
        Az = self.eval_poly_at(self.A, z)

        inverses = self.multi_inv([z - x for x in self.DOMAIN])

        for i, x in enumerate(inverses):
            r.append(Az * self.Aprime_DOMAIN_inv[i] * x % self.MODULUS)

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
                q[i] = (f[i] - y) * self.INVERSES[i - index] % self.MODULUS
                q[index] += (f[i] - y) * self.INVERSES[index - i] * self.Aprime_DOMAIN[index] * self.Aprime_DOMAIN_inv[i] % self.MODULUS

        q[index] %= self.MODULUS

        return q


    def compute_outer_quotient_in_evaluation_form(self, f, z, y):
        """
        Compute the quotient q(X) = (f(X) - y)) / (X - z) in evaluation form. Note that this only works if the quotient
        is exact, i.e. f(z) = y, and otherwise returns garbage
        """
        q = [0] * self.WIDTH
        for i in range(self.WIDTH):
            q[i] = self.div(f[i] - y, self.DOMAIN[i] - z)

        return q

    def add(self, x, y):
        return (x+y) % self.MODULUS

    def sub(self, x, y):
        return (x-y) % self.MODULUS

    def mul(self, x, y):
        return (x*y) % self.MODULUS

    def exp(self, x, p):
        return pow(x, p, self.MODULUS)

    # Modular inverse using the extended Euclidean algorithm
    def inv(self, a):
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % self.MODULUS, self.MODULUS
        while low > 1:
            r = high//low
            nm, new = hm-lm*r, high-low*r
            lm, low, hm, high = nm, new, lm, low
        return lm % self.MODULUS

    def multi_inv(self, values):
        partials = [1]
        for i in range(len(values)):
            partials.append(self.mul(partials[-1], values[i] or 1))
        inv = self.inv(partials[-1])
        outputs = [0] * len(values)
        for i in range(len(values), 0, -1):
            outputs[i-1] = self.mul(partials[i-1], inv) if values[i-1] else 0
            inv = self.mul(inv, values[i-1] or 1)
        return outputs

    def div(self, x, y):
        return self.mul(x, self.inv(y))

    # Evaluate a polynomial at a point
    def eval_poly_at(self, p, x):
        y = 0
        power_of_x = 1
        for i, p_coeff in enumerate(p):
            y += power_of_x * p_coeff
            power_of_x = (power_of_x * x) % self.MODULUS
        return y % self.MODULUS
        
    # Arithmetic for polynomials
    def add_polys(self, a, b):
        return [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0))
                % self.MODULUS for i in range(max(len(a), len(b)))]

    def sub_polys(self, a, b):
        return [((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0))
                % self.MODULUS for i in range(max(len(a), len(b)))]
    
    def mul_by_const(self, a, c):
        return [(x*c) % self.MODULUS for x in a]
    
    def mul_polys(self, a, b):
        o = [0] * (len(a) + len(b) - 1)
        for i, aval in enumerate(a):
            for j, bval in enumerate(b):
                o[i+j] += a[i] * b[j]
        return [x % self.MODULUS for x in o]
    
    def div_polys(self, a, b):
        assert len(a) >= len(b)
        a = [x for x in a]
        o = []
        apos = len(a) - 1
        bpos = len(b) - 1
        diff = apos - bpos
        while diff >= 0:
            quot = self.div(a[apos], b[bpos])
            o.insert(0, quot)
            for i in range(bpos, -1, -1):
                a[diff+i] -= b[i] * quot
            apos -= 1
            diff -= 1
        return [x % self.MODULUS for x in o]

    def mod_polys(self, a, b):
        return self.sub_polys(a, self.mul_polys(b, self.div_polys(a, b)))[:len(b)-1]

    # Build a polynomial from a few coefficients
    def sparse(self, coeff_dict):
        o = [0] * (max(coeff_dict.keys()) + 1)
        for k, v in coeff_dict.items():
            o[k] = v % self.MODULUS
        return o

    # Build a polynomial that returns 0 at all specified xs
    def zpoly(self, xs):
        root = [1]
        for x in xs:
            root.insert(0, 0)
            for j in range(len(root)-1):
                root[j] -= root[j+1] * x
        return [x % self.MODULUS for x in root]
    
    # Given p+1 y values and x values with no errors, recovers the original
    # p+1 degree polynomial.
    # Lagrange interpolation works roughly in the following way.
    # 1. Suppose you have a set of points, eg. x = [1, 2, 3], y = [2, 5, 10]
    # 2. For each x, generate a polynomial which equals its corresponding
    #    y coordinate at that point and 0 at all other points provided.
    # 3. Add these polynomials together.
    
    def lagrange_interp(self, xs, ys):
        # Generate master numerator polynomial, eg. (x - x1) * (x - x2) * ... * (x - xn)
        root = self.zpoly(xs)
        assert len(root) == len(ys) + 1
        # print(root)
        # Generate per-value numerator polynomials, eg. for x=x2,
        # (x - x1) * (x - x3) * ... * (x - xn), by dividing the master
        # polynomial back by each x coordinate
        nums = [self.div_polys(root, [-x, 1]) for x in xs]
        # Generate denominators by evaluating numerator polys at each x
        denoms = [self.eval_poly_at(nums[i], xs[i]) for i in range(len(xs))]
        invdenoms = self.multi_inv(denoms)
        # Generate output polynomial, which is the sum of the per-value numerator
        # polynomials rescaled to have the right y values
        b = [0 for y in ys]
        for i in range(len(xs)):
            yslice = self.mul(ys[i], invdenoms[i])
            for j in range(len(ys)):
                if nums[i][j] and ys[i]:
                    b[j] += nums[i][j] * yslice
        return [x % self.MODULUS for x in b]

    # Optimized poly evaluation for degree 4
    def eval_quartic(self, p, x):
        xsq = x * x % self.MODULUS
        xcb = xsq * x
        return (p[0] + p[1] * x + p[2] * xsq + p[3] * xcb) % self.MODULUS

    # Optimized version of the above restricted to deg-4 polynomials
    def lagrange_interp_4(self, xs, ys):
        x01, x02, x03, x12, x13, x23 = \
            xs[0] * xs[1], xs[0] * xs[2], xs[0] * xs[3], xs[1] * xs[2], xs[1] * xs[3], xs[2] * xs[3]
        m = self.MODULUS
        eq0 = [-x12 * xs[3] % m, (x12 + x13 + x23), -xs[1]-xs[2]-xs[3], 1]
        eq1 = [-x02 * xs[3] % m, (x02 + x03 + x23), -xs[0]-xs[2]-xs[3], 1]
        eq2 = [-x01 * xs[3] % m, (x01 + x03 + x13), -xs[0]-xs[1]-xs[3], 1]
        eq3 = [-x01 * xs[2] % m, (x01 + x02 + x12), -xs[0]-xs[1]-xs[2], 1]
        e0 = self.eval_poly_at(eq0, xs[0])
        e1 = self.eval_poly_at(eq1, xs[1])
        e2 = self.eval_poly_at(eq2, xs[2])
        e3 = self.eval_poly_at(eq3, xs[3])
        e01 = e0 * e1
        e23 = e2 * e3
        invall = self.inv(e01 * e23)
        inv_y0 = ys[0] * invall * e1 * e23 % m
        inv_y1 = ys[1] * invall * e0 * e23 % m
        inv_y2 = ys[2] * invall * e01 * e3 % m
        inv_y3 = ys[3] * invall * e01 * e2 % m
        return [(eq0[i] * inv_y0 + eq1[i] * inv_y1 + eq2[i] * inv_y2 + eq3[i] * inv_y3) % m for i in range(4)]
    
    # Optimized version of the above restricted to deg-2 polynomials
    def lagrange_interp_2(self, xs, ys):
        m = self.MODULUS
        eq0 = [-xs[1] % m, 1]
        eq1 = [-xs[0] % m, 1]
        e0 = self.eval_poly_at(eq0, xs[0])
        e1 = self.eval_poly_at(eq1, xs[1])
        invall = self.inv(e0 * e1)
        inv_y0 = ys[0] * invall * e1
        inv_y1 = ys[1] * invall * e0
        return [(eq0[i] * inv_y0 + eq1[i] * inv_y1) % m for i in range(2)]

    # Optimized version of the above restricted to deg-4 polynomials
    def multi_interp_4(self, xsets, ysets):
        data = []
        invtargets = []
        for xs, ys in zip(xsets, ysets):
            x01, x02, x03, x12, x13, x23 = \
                xs[0] * xs[1], xs[0] * xs[2], xs[0] * xs[3], xs[1] * xs[2], xs[1] * xs[3], xs[2] * xs[3]
            m = self.MODULUS
            eq0 = [-x12 * xs[3] % m, (x12 + x13 + x23), -xs[1]-xs[2]-xs[3], 1]
            eq1 = [-x02 * xs[3] % m, (x02 + x03 + x23), -xs[0]-xs[2]-xs[3], 1]
            eq2 = [-x01 * xs[3] % m, (x01 + x03 + x13), -xs[0]-xs[1]-xs[3], 1]
            eq3 = [-x01 * xs[2] % m, (x01 + x02 + x12), -xs[0]-xs[1]-xs[2], 1]
            e0 = self.eval_quartic(eq0, xs[0])
            e1 = self.eval_quartic(eq1, xs[1])
            e2 = self.eval_quartic(eq2, xs[2])
            e3 = self.eval_quartic(eq3, xs[3])
            data.append([ys, eq0, eq1, eq2, eq3])
            invtargets.extend([e0, e1, e2, e3])
        invalls = self.multi_inv(invtargets)
        o = []
        for (i, (ys, eq0, eq1, eq2, eq3)) in enumerate(data):
            invallz = invalls[i*4:i*4+4]
            inv_y0 = ys[0] * invallz[0] % m
            inv_y1 = ys[1] * invallz[1] % m
            inv_y2 = ys[2] * invallz[2] % m
            inv_y3 = ys[3] * invallz[3] % m
            o.append([(eq0[i] * inv_y0 + eq1[i] * inv_y1 + eq2[i] * inv_y2 + eq3[i] * inv_y3) % m for i in range(4)])
        # assert o == [self.lagrange_interp_4(xs, ys) for xs, ys in zip(xsets, ysets)]
        return o


if __name__ == "__main__":
    primefield = PrimeField(11, 4)
    for i, x in enumerate(primefield.DOMAIN):
        assert primefield.eval_poly_at(primefield.lagrange_polys[i], x) == 1
        for y in primefield.DOMAIN[:i] + primefield.DOMAIN[i+1:]:
            assert primefield.eval_poly_at(primefield.lagrange_polys[i], y) == 0

    poly = [1, 2]
    poly_eval = [primefield.eval_poly_at(poly, x) for x in primefield.DOMAIN]


    assert primefield.eval_poly_at(poly, 5) == primefield.evaluate_polynomial_in_evaluation_form(poly_eval, 5)

    poly_eval_quotient = primefield.compute_inner_quotient_in_evaluation_form(poly_eval, 2)

    poly_quotient = primefield.div_polys([poly[0] - poly_eval[2]] + poly[1:], [-2, 1])

    poly_quotient_eval = [primefield.eval_poly_at(poly_quotient, x) for x in primefield.DOMAIN]

    assert poly_eval_quotient == poly_quotient_eval

    y = primefield.eval_poly_at(poly, 7)

    poly_eval_quotient2 = primefield.compute_outer_quotient_in_evaluation_form(poly_eval, 7, y)
    
    poly_quotient2 = primefield.div_polys([poly[0] - y] + poly[1:], [-7, 1])

    poly_quotient_eval2 = [primefield.eval_poly_at(poly_quotient2, x) for x in primefield.DOMAIN]

    assert poly_eval_quotient2 == poly_quotient_eval2