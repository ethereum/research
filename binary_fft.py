def log2(x):
    return 0 if x <= 1 else 1 + log2(x // 2)

def raw_mul(a, b):
    if a*b == 0:
        return 0
    o = 0
    for i in range(log2(b) + 1):
        if b & (1<<i):
            o ^= a<<i
    return o

def raw_mod(a, b):
    blog = log2(b)
    alog = log2(a)
    while alog >= blog:
        if a & (1<<alog):
            a ^= (b << (alog - blog))
        alog -= 1
    return a

class BinaryField():
    def __init__(self, modulus):
        self.modulus = modulus
        self.height = log2(self.modulus)
        self.order = 2**self.height - 1
        for base in range(2, modulus - 1):
            powers = [1]
            while (len(powers) == 1 or powers[-1] != 1) and len(powers) < self.order + 2:
                powers.append(raw_mod(raw_mul(powers[-1], base), self.modulus))
            powers.pop()
            if len(powers) == self.order:
                self.cache = powers
                self.invcache = [None] * (self.order + 1)
                for i, p in enumerate(powers):
                    self.invcache[p] = i
                return
        raise Exception("Bad modulus")

    def add(self, x, y):
        return x ^ y

    sub = add

    def mul(self, x, y):
        return 0 if x*y == 0 else self.cache[(self.invcache[x] + self.invcache[y]) % self.order]

    def sqr(self, x):
        return 0 if x == 0 else self.cache[(self.invcache[x] * 2) % self.order]

    def div(self, x, y):
        return 0 if x == 0 else self.cache[(self.invcache[x] - self.invcache[y]) % self.order]

    def inv(self, x):
        return self.cache[(self.order - self.invcache[x]) % self.order]

    def exp(self, x, p):
        return 1 if p == 0 else 0 if x == 0 else self.cache[(self.invcache[x] * p) % self.order]

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
            y ^= self.mul(power_of_x, p_coeff)
            power_of_x = self.mul(power_of_x, x)
        return y
        
    # Arithmetic for polynomials
    def add_polys(self, a, b):
        return [((a[i] if i < len(a) else 0) ^ (b[i] if i < len(b) else 0))
                for i in range(max(len(a), len(b)))]

    sub_polys = add_polys
    
    def mul_by_const(self, a, c):
        return [self.mul(x, c) for x in a]
    
    def mul_polys(self, a, b):
        o = [0] * (len(a) + len(b) - 1)
        for i, aval in enumerate(a):
            for j, bval in enumerate(b):
                o[i+j] ^= self.mul(a[i], b[j])
        return o
    
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
                a[diff+i] ^= self.mul(b[i], quot)
            apos -= 1
            diff -= 1
        return o

    # Build a polynomial that returns 0 at all specified xs
    def zpoly(self, xs):
        root = [1]
        for x in xs:
            root.insert(0, 0)
            for j in range(len(root)-1):
                root[j] ^= self.mul(root[j+1], x)
        return root
    
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
        nums = [self.div_polys(root, [x, 1]) for x in xs]
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
                    b[j] ^= self.mul(nums[i][j], yslice)
        return b

def _simple_ft(field, vals):
    assert len(vals) == 2**field.height
    return [field.eval_poly_at(vals, i) for i in range(2**field.height)]

# Returns `evens` and `odds` such that:
# poly(x) = evens(x^2+kx) + x * odds(x^2+kx)
# poly(x+k) = evens(x^2+kx) + (x+k) * odds(x^2+kx)
# 
# Note that this satisfies two other invariants:
#
# poly(x+k) - poly(x) = k * odds(x^2+kx)
# poly(x)*(x+k) - poly(x+k)*x = k * evens(x^2+kx)

def cast(field, poly, k):
    if len(poly) <= 2:
        return ([poly[0]], [poly[1] if len(poly) == 2 else 0])
    mod_power = 2
    while mod_power * 2 < len(poly):
        mod_power *= 2
    half_mod_power = mod_power // 2
    k_to_half_mod_power = field.exp(k, half_mod_power)
    low = poly + [0] * (mod_power * 2 - len(poly))
    high = low[len(low)-half_mod_power:]
    low = low[:len(low)-mod_power] + [low[i] ^ field.mul(low[i+half_mod_power], k_to_half_mod_power) for i in range(len(low)-mod_power, len(low)-half_mod_power)]
    high = low[len(low)-half_mod_power:] + high
    low = low[:len(low)-mod_power] + [low[i] ^ field.mul(low[i+half_mod_power], k_to_half_mod_power) for i in range(len(low)-mod_power, len(low)-half_mod_power)]
    low_cast = cast(field, low, k) 
    high_cast = cast(field, high, k)
    return (low_cast[0] + high_cast[0], low_cast[1] + high_cast[1])

# Returns a polynomial p2 such that p2(x) = poly(x^2+kx)
def compose(field, poly, k):
    if len(poly) == 1:
        return poly + [0]
    mod_power = 1
    while mod_power * 2 < len(poly):
        mod_power *= 2
    k_to_mod_power = field.exp(k, mod_power)
    low = compose(field, poly[:mod_power], k) + [0] * mod_power * 3
    high = compose(field, poly[mod_power:], k) + [0] * mod_power * 3
    return [low[i] ^ field.mul(high[i-mod_power], k_to_mod_power) ^ high[i-2*mod_power] for i in range(mod_power*4)]

# Equivalent to [field.eval_poly_at(poly, x) for x in domain]
def fft(field, poly, domain):
    # Base case: constant polynomials
    if len(domain) == 1:
        return [poly[0]]
    # Split the domain into two cosets A and B, where for x in A, x+offset is in B
    offset = domain[1]
    # Get evens, odds such that:
    # poly(x) = evens(x^2+offset*x) + x * odds(x^2+offset*x)
    # poly(x+k) = evens(x^2+offset*x) + (x+k) * odds(x^2+offset*x)
    evens, odds = cast(field, poly, offset)
    # The smaller domain D = [x**2 - offset*x for x in A] = [x**2 - offset*x for x in B]
    casted_domain = [field.mul(x, offset ^ x) for x in domain][::2]
    # Two half-size sub-problems over the smaller domain, recovering
    # evaluations of evens and odds over the smaller domain
    even_points = fft(field, evens, casted_domain)
    odd_points = fft(field, odds, casted_domain)
    # Combine the evaluations of evens and odds into evaluations of poly
    L = [e ^ field.mul(d, o) for d,e,o in zip(domain[::2], even_points, odd_points)]
    R = [e ^ field.mul(d, o) for d,e,o in zip(domain[1::2], even_points, odd_points)]
    return [R[i//2] if i%2 else L[i//2] for i in range(len(domain))]

# The inverse function of fft, does the steps backwards
def invfft(field, vals, domain):
    # Base case: constant polynomials
    if len(domain) == 1:
        return [vals[0]]
    # Split the domain into two cosets A and B, where for x in A, x+offset is in B
    offset = domain[1]
    # Compute the evaluations of the evens and odds polynomials using the invariants:
    # poly(x+k) - poly(x) = k * odds(x^2+kx)
    # poly(x)*(x+k) - poly(x+k)*x = k * evens(x^2+kx)
    L, R = vals[::2], vals[1::2]
    even_points = [field.div(field.mul(l, d ^ offset) ^ field.mul(r, d), offset) for d, l, r in zip(domain[::2], L, R)]
    odd_points = [field.div(l ^ r, offset) for d, l, r in zip(domain[::2], L, R)]
    # The smaller domain D = [x**2 - offset*x for x in A] = [x**2 - offset*x for x in B]
    casted_domain = [field.mul(x, offset ^ x) for x in domain][::2]
    # Two half-size problems over the smaller domains, recovering
    # the polynomials evens and odds
    evens = invfft(field, even_points, casted_domain)
    odds = invfft(field, odd_points, casted_domain)
    # Given evens and odds where poly(x) = evens(x^2+offset*x) + x * odds(x^2+offset*x),
    # recover poly
    composed_evens = compose(field, evens, offset) + [0]
    composed_odds = compose(field, odds, offset) + [0]
    o = [composed_evens[i] ^ composed_odds[i-1] for i in range(len(vals))]
    return o
