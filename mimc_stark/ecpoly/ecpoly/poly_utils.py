class PrimeField():
    def __init__(self, modulus):
        assert pow(2, modulus, modulus) == 2
        self.modulus = modulus

    def add(self, x, y):
        return (x+y) % self.modulus

    def sub(self, x, y):
        return (x-y) % self.modulus

    def mul(self, x, y):
        return (x*y) % self.modulus

    def inv(self, a):
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % self.modulus, self.modulus
        while low > 1:
            r = high//low
            nm, new = hm-lm*r, high-low*r
            lm, low, hm, high = nm, new, lm, low
        return lm % self.modulus

    def div(self, x, y):
        if x == 0 and y == 0:
            return 1
        return self.mul(x, self.inv(y))

    # Evaluate a polynomial at a point
    def eval_poly_at(self, p, x):
        if x == 0:
            return p[0]
        y = 0
        power_of_x = 1
        for i, p_coeff in enumerate(p):
            y += power_of_x * p_coeff
            power_of_x = (power_of_x * x) % self.modulus
        return y % self.modulus

    # Build a polynomial that returns 0 at all xs
    def zpoly(self, xs):
        root = [1]
        for x in xs:
            root.insert(0, 0)
            for j in range(len(root)-1):
                root[j] -= root[j+1] * x
        return [x % self.modulus for x in root]
    
    # Given p+1 y values and x values with no errors, recovers the original
    # p+1 degree polynomial.
    # Lagrange interpolation works roughly in the following way.
    # 1. Suppose you have a set of points, eg. x = [1, 2, 3], y = [2, 5, 10]
    # 2. For each x, generate a polynomial which equals its corresponding
    #    y coordinate at that point and 0 at all other points provided.
    # 3. Add these polynomials together.
    
    def lagrange_interp(self, pieces, xs):
        # Generate master numerator polynomial, eg. (x - x1) * (x - x2) * ... * (x - xn)
        root = self.zpoly(xs)
        #print(root)
        assert len(root) == len(pieces) + 1
        # print(root)
        # Generate per-value numerator polynomials, eg. for x=x2,
        # (x - x1) * (x - x3) * ... * (x - xn), by dividing the master
        # polynomial back by each x coordinate
        nums = []
        for x in xs:
            output = [0] * (len(root) - 2) + [1]
            for j in range(len(root) - 2, 0, -1):
                output[j-1] = root[j] + output[j] * x
            assert len(output) == len(pieces)
            nums.append(output)
        #print(nums)
        # Generate denominators by evaluating numerator polys at each x
        denoms = [self.eval_poly_at(nums[i], xs[i]) for i in range(len(xs))]
        # Generate output polynomial, which is the sum of the per-value numerator
        # polynomials rescaled to have the right y values
        b = [0 for p in pieces]
        for i in range(len(xs)):
            yslice = self.div(pieces[i], denoms[i])
            for j in range(len(pieces)):
                if nums[i][j] and pieces[i]:
                    b[j] += nums[i][j] * yslice
        return [x % self.modulus for x in b]
    
    def add_polys(self, a, b):
        return [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0))
                % self.modulus for i in range(max(len(a), len(b)))]

    def sub_polys(self, a, b):
        return [((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0))
                % self.modulus for i in range(max(len(a), len(b)))]
    
    def mul_by_const(self, a, c):
        return [(x*c) % self.modulus for x in a]
    
    def mul_polys(self, a, b):
        o = [0] * (len(a) + len(b) - 1)
        for i, aval in enumerate(a):
            for j, bval in enumerate(b):
                o[i+j] += a[i] * b[j]
        return [x % self.modulus for x in o]
    
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
        return [x % self.modulus for x in o]
    
    def compose_polys(self, a, b):
        o = []
        p = [1]
        for c in a:
            o = self.add_polys(o, self.mul_by_const(p, c))
            p = self.mul_polys(p, b)
        return o
    
