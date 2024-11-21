def log2(x):
    o = 0
    while x > 1:
        x //= 2
        o += 1
    return o

def is_power_of_2(x):
    return x > 0 and x&(x-1) == 0

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
        for base in range(2, min(modulus - 1, 80)):
            powers = [1]
            while (len(powers) == 1 or powers[-1] != 1) and len(powers) < self.order + 2:
                powers.append(raw_mod(raw_mul(powers[-1], base), self.modulus))
            powers.pop()
            if len(powers) == self.order:
                self.cache = powers + powers
                self.invcache = [None] * (self.order + 1)
                for i, p in enumerate(powers):
                    self.invcache[p] = i
                return
        raise Exception("Bad modulus")

    def add(self, x, y):
        return x ^ y

    sub = add

    def mul(self, x, y):
        return 0 if x*y == 0 else self.cache[self.invcache[x] + self.invcache[y]]

    def sqr(self, x):
        return 0 if x == 0 else self.cache[(self.invcache[x] * 2) % self.order]

    def div(self, x, y):
        return 0 if x == 0 else self.cache[self.invcache[x] + self.order - self.invcache[y]]

    def inv(self, x):
        assert x != 0
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

def _simple_ft(field, domain, poly):
    return [field.eval_poly_at(poly, i) for i in domain]

# Returns `evens` and `odds` such that:
# poly(x) = evens(x**2+kx) + x * odds(x**2+kx)
# poly(x+k) = evens(x**2+kx) + (x+k) * odds(x**2+kx)
# 
# Note that this satisfies two other invariants:
#
# poly(x+k) - poly(x) = k * odds(x**2+kx)
# poly(x)*(x+k) - poly(x+k)*x = k * evens(x**2+kx)

def cast(field, poly, k):
    if len(poly) <= 2:
        return ([poly[0]], [poly[1] if len(poly) == 2 else 0])
    assert is_power_of_2(len(poly))
    mod_power = len(poly)//2
    half_mod_power = mod_power // 2
    k_to_half_mod_power = field.exp(k, half_mod_power)
    # Calculate low = poly % (x**2 - k*x)**half_mod_power
    # and high = poly // (x**2 - k*x)**half_mod_power
    # Note that (x**2 - k*x)**n = x**2n - k**n * x**n in binary fields
    low_and_high = poly[::]
    for i in range(mod_power, half_mod_power*3):
        low_and_high[i] ^= field.mul(low_and_high[i+half_mod_power], k_to_half_mod_power)
    for i in range(half_mod_power, mod_power):
        low_and_high[i] ^= field.mul(low_and_high[i+half_mod_power], k_to_half_mod_power)
    # Recursively compute two half-size sub-problems, low and high
    low_cast = cast(field, low_and_high[:mod_power], k) 
    high_cast = cast(field, low_and_high[mod_power:], k)
    # Combine the results
    return (low_cast[0] + high_cast[0], low_cast[1] + high_cast[1])

# Returns a polynomial p2 such that p2(x) = poly(x**2+kx)
def compose(field, poly, k):
    if len(poly) == 2:
        return [poly[0], field.mul(poly[1], k), poly[1], 0]
    if len(poly) == 1:
        return poly + [0]
    # Largest mod_power=2**k such that mod_power >= len(poly)/2
    assert is_power_of_2(len(poly))
    mod_power = len(poly)//2
    k_to_mod_power = field.exp(k, mod_power)
    # Recursively compute two half-size sub-problems, the bottom and top half
    # of the polynomial
    low = compose(field, poly[:mod_power], k)
    high = compose(field, poly[mod_power:], k)
    # Combine them together, multiplying the top one by (x**2-k*x)**n
    # Note that (x**2 - k*x)**n = x**2n - k**n * x**n in binary fields
    o = [0] * len(poly) * 2
    for i, (L, H) in enumerate(zip(low, high)):
        o[i] ^= L
        o[i+mod_power] ^= field.mul(H, k_to_mod_power)
        o[i+2*mod_power] ^= H
    return o

# Equivalent to [field.eval_poly_at(poly, x) for x in domain]
# Special thanks to www.math.clemson.edu/~sgao/papers/GM10.pdf for insights
# though this algorithm is not exactly identical to any algorithm in the paper
def fft(field, domain, poly):
    # Base case: constant polynomials
    # if len(domain) == 1:
    #     return [poly[0]]
    if len(domain) <= 8:
        return _simple_ft(field, domain, poly)
    # Split the domain into two cosets A and B, where for x in A, x+offset is in B
    offset = domain[1]
    # Get evens, odds such that:
    # poly(x) = evens(x**2+offset*x) + x * odds(x**2+offset*x)
    # poly(x+k) = evens(x**2+offset*x) + (x+k) * odds(x**2+offset*x)
    evens, odds = cast(field, poly, offset)
    # The smaller domain D = [x**2 - offset*x for x in A] = [x**2 - offset*x for x in B]
    casted_domain = [field.mul(x, offset ^ x) for x in domain[::2]]
    # Two half-size sub-problems over the smaller domain, recovering
    # evaluations of evens and odds over the smaller domain
    even_points = fft(field, casted_domain, evens)
    odd_points = fft(field, casted_domain, odds)
    # Combine the evaluations of evens and odds into evaluations of poly
    o = []
    for i in range(len(domain)//2):
        o.append(even_points[i] ^ field.mul(domain[i*2], odd_points[i]))
        o.append(even_points[i] ^ field.mul(domain[i*2+1], odd_points[i]))
    return o

# The inverse function of fft, does the steps backwards
def invfft(field, domain, vals):
    # Base case: constant polynomials
    if len(domain) == 1:
        return [vals[0]]
    # if len(domain) <= 4:
    #     return field.lagrange_interp(domain, vals)
    # Split the domain into two cosets A and B, where for x in A, x+offset is in B
    offset = domain[1]
    # Compute the evaluations of the evens and odds polynomials using the invariants:
    # poly(x+k) - poly(x) = k * odds(x**2+kx)
    # poly(x)*(x+k) - poly(x+k)*x = k * evens(x**2+kx)
    even_points = [0] * (len(vals)//2)
    odd_points = [0] * (len(vals)//2)
    for i in range(len(domain)//2):
        p_of_x, p_of_x_plus_k = vals[i*2], vals[i*2+1]
        x = domain[i*2]
        even_points[i] = field.div(field.mul(p_of_x, x ^ offset) ^ field.mul(p_of_x_plus_k, x), offset)
        odd_points[i] = field.div(p_of_x ^ p_of_x_plus_k, offset)
    casted_domain = [field.mul(x, offset ^ x) for x in domain[::2]]
    # Two half-size problems over the smaller domains, recovering
    # the polynomials evens and odds
    evens = invfft(field, casted_domain, even_points)
    odds = invfft(field, casted_domain, odd_points)
    # Given evens and odds where poly(x) = evens(x**2+offset*x) + x * odds(x**2+offset*x),
    # recover poly
    composed_evens = compose(field, evens, offset) + [0]
    composed_odds = [0] + compose(field, odds, offset)
    o = [composed_evens[i] ^ composed_odds[i] for i in range(len(vals))]
    return o

# shift_polys[i][j] is the 2**j degree coefficient of the polynomial that evaluates to [1,1...1, 0,0....0] with 2**(i-1) ones and 2**(i-1) zeroes
shift_polys = [[], [1], [32755, 32755], [52774, 60631, 8945], [38902, 5560, 44524, 12194], [55266, 46488, 60321, 5401, 40130], [21827, 32224, 51565, 15072, 8277, 64379], [59460, 15452, 60370, 24737, 20321, 35516, 39606], [42623, 56997, 25925, 15351, 16625, 47045, 38250, 17462], [7575, 27410, 32434, 22187, 28933, 15447, 37964, 38186, 4776], [39976, 61188, 42456, 2155, 6178, 34033, 52305, 14913, 2896, 48908], [6990, 12021, 36054, 16198, 17011, 14018, 58553, 13272, 25318, 5288, 21429], [16440, 34925, 14360, 22561, 43883, 36645, 7613, 26531, 8597, 59502, 61283, 53412]]

def invfft2(field, vals):
    if len(vals) == 1:
        return [vals[0]]
    L = invfft2(field, vals[:len(vals)//2])
    R = shift(field, invfft2(field, vals[len(vals)//2:]), len(vals)//2)
    o = [0] * len(vals)
    for j, (l, r) in enumerate(zip(L, R)):
        o[j] ^= l
        for i, coeff in enumerate(shift_polys[log2(len(vals))]):
            o[2**i+j] ^= field.mul(l ^ r, coeff)
    # print(vals, o)
    return o

# def invfft(field, domain, vals): return invfft2(field, vals)

# Multiplies two polynomials using the FFT method
def mul(field, domain, p1, p2):
    assert len(p1) <= len(domain) and len(p2) <= len(domain)
    values1 = fft(field, domain, p1)
    values2 = fft(field, domain, p2)
    values3 = [field.mul(v1, v2) for v1, v2 in zip(values1, values2)]
    return invfft(field, domain, values3)

# Generates the polynomial `p(x) = (x - xs[0]) * (x - xs[1]) * ...`
def zpoly(field, xs):
    if len(xs) == 0:
        # print([1], domain, xs)
        return [1]
    if len(xs) == 1:
        # print([xs[0], 1], domain, xs)
        return [xs[0], 1]
    domain = list(range(2**log2(max(xs)+1) * 2))
    offset = domain[1]
    zL = zpoly(field, xs[::2])
    zR = zpoly(field, xs[1::2])
    o = mul(field, domain, zL, zR)
    # print(o, domain, xs)
    return o

# Returns q(x) = p(x + k)
def shift(field, poly, k):
    if len(poly) == 1:
        return poly
    # Largest mod_power=2**k such that mod_power >= len(poly)/2
    assert is_power_of_2(len(poly))
    mod_power = len(poly)//2
    k_to_mod_power = field.exp(k, mod_power)
    # Calculate low = poly % (x+k)**mod_power
    # and high = poly // (x+k)**mod_power
    # Note that (x+k)**n = x**n + k**n for power-of-two powers in binary fields
    low_and_high = poly[::]
    for i in range(mod_power):
        low_and_high[i] ^= field.mul(low_and_high[i+mod_power], k_to_mod_power)
    return shift(field, low_and_high[:mod_power], k) + shift(field, low_and_high[mod_power:], k)

# Interpolates the polynomial where `p(xs[i]) = vals[i]`
def interpolate(field, xs, vals):
    domain_size = 2**(log2(max(xs)) + 1)
    assert domain_size * 2 <= 2**field.height
    domain = list(range(domain_size))
    big_domain = list(range(domain_size * 2))
    z = zpoly(field, [x for x in domain if x not in xs])
    # print("z = ", z)
    z_values = fft(field, big_domain, z)
    # print("z_values = ", z_values)
    p_times_z_values = [0] * len(domain)
    for v, d in zip(vals, xs):
        p_times_z_values[d] = field.mul(v, z_values[d])
    # print("p_times_z_values = ", p_times_z_values)
    p_times_z = invfft(field, domain, p_times_z_values)
    # print("p_times_z = ", p_times_z)
    shifted_p_times_z_values = fft(field, big_domain, p_times_z)[domain_size:]
    # print("shifted_p_times_z_values =", shifted_p_times_z_values)
    shifted_p_values = [field.div(x, y) for x,y in zip(shifted_p_times_z_values, z_values[domain_size:])]
    # print("shifted_p_values =", shifted_p_values)
    shifted_p = invfft(field, domain, shifted_p_values)
    return shift(field, shifted_p, domain_size)
