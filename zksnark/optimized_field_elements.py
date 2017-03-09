field_modulus = 21888242871839275222246405745257275088696311157297823662689037894645226208583
FQ12_modulus_coeffs = [82, 0, 0, 0, 0, 0, -18, 0, 0, 0, 0, 0] # Implied + [1]
FQ12_mc_tuples = [(i, c) for i, c in enumerate(FQ12_modulus_coeffs) if c]

# python3 compatibility
try:
    foo = long
except:
    long = int

# Extended euclidean algorithm to find modular inverses for
# integers
def prime_field_inv(a, n):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % n, n
    while low > 1:
        r = high//low
        nm, new = hm-lm*r, high-low*r
        lm, low, hm, high = nm, new, lm, low
    return lm % n

# Utility methods for polynomial math
def deg(p):
    d = len(p) - 1
    while p[d] == 0 and d:
        d -= 1
    return d

def poly_rounded_div(a, b):
    dega = deg(a)
    degb = deg(b)
    temp = [x for x in a]
    o = [0 for x in a]
    for i in range(dega - degb, -1, -1):
        o[i] = (o[i] + temp[degb + i] * prime_field_inv(b[degb], field_modulus))
        for c in range(degb + 1):
            temp[c + i] = (temp[c + i] - o[c])
    return [x % field_modulus for x in o[:deg(o)+1]]

def karatsuba(a, b, c, d):
    L = len(a)
    EXTENDED_LEN = L * 2 - 1
    # phi = (a+b)(c+d)
    # psi = (a-b)(c-d)
    phi, psi, bd2 = [0] * EXTENDED_LEN, [0] * EXTENDED_LEN, [0] * EXTENDED_LEN
    for i in range(L):
        for j in range(L):
            phi[i + j] += (a[i] + b[i]) * (c[j] + d[j])
            psi[i + j] += (a[i] - b[i]) * (c[j] - d[j])
            bd2[i + j] += b[i] * d[j] * 2
    o = [0] * (L * 4 - 1)
    # L = (phi + psi - bd2) / 2
    # M = (phi - psi) / 2
    # H = bd2 / 2
    for i in range(L * 2 - 1):
        o[i] += phi[i] + psi[i] - bd2[i]
        o[i + L] += phi[i] - psi[i]
        o[i + L * 2] += bd2[i]
    inv_2 = (field_modulus + 1) // 2
    return [a * inv_2 if a % 2 else a // 2 for a in o]

o = karatsuba([1, 3], [3, 1], [1, 3], [3, 1])
assert [x % field_modulus for x in o] == [1, 6, 15, 20, 15, 6, 1]

# A class for elements in polynomial extension fields
class FQP():
    def __init__(self, coeffs, modulus_coeffs): 
        assert len(coeffs) == len(modulus_coeffs)
        self.coeffs = coeffs
        # The coefficients of the modulus, without the leading [1]
        self.modulus_coeffs = modulus_coeffs
        # The degree of the extension field
        self.degree = len(self.modulus_coeffs)

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__([(x+y) % field_modulus for x,y in zip(self.coeffs, other.coeffs)])

    def __sub__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__([(x-y) % field_modulus for x,y in zip(self.coeffs, other.coeffs)])

    def __mul__(self, other):
        if isinstance(other, (int, long)):
            return self.__class__([c * other % field_modulus for c in self.coeffs])
        else:
            # assert isinstance(other, self.__class__)
            b = [0] * (self.degree * 2 - 1)
            inner_enumerate = list(enumerate(other.coeffs))
            for i, eli in enumerate(self.coeffs):
                for j, elj in inner_enumerate:
                    b[i + j] += eli * elj
            # MID = len(self.coeffs) // 2
            # b = karatsuba(self.coeffs[:MID], self.coeffs[MID:], other.coeffs[:MID], other.coeffs[MID:])
            for exp in range(self.degree - 2, -1, -1):
                top = b.pop()
                for i, c in self.mc_tuples:
                    b[exp + i] -= top * c
            return self.__class__([x % field_modulus for x in b])

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, (int, long)):
            return self.__class__([c * prime_field_inv(other, field_modulus) % field_modulus for c in self.coeffs])
        else:
            assert isinstance(other, self.__class__)
            return self * other.inv()

    def __truediv__(self, other):
        return self.__div__(other)

    def __pow__(self, other):
        o = self.__class__([1] + [0] * (self.degree - 1))
        t = self
        while other > 0:
            if other & 1:
                o = o * t
            other >>= 1
            t = t * t
        return o

    # Extended euclidean algorithm used to find the modular inverse
    def inv(self):
        lm, hm = [1] + [0] * self.degree, [0] * (self.degree + 1)
        low, high = self.coeffs + [0], self.modulus_coeffs + [1]
        while deg(low):
            r = poly_rounded_div(high, low)
            r += [0] * (self.degree + 1 - len(r))
            nm = [x for x in hm]
            new = [x for x in high]
            # assert len(lm) == len(hm) == len(low) == len(high) == len(nm) == len(new) == self.degree + 1
            for i in range(self.degree + 1):
                for j in range(self.degree + 1 - i):
                    nm[i+j] -= lm[i] * r[j]
                    new[i+j] -= low[i] * r[j]
            nm = [x % field_modulus for x in nm]
            new = [x % field_modulus for x in new]
            lm, low, hm, high = nm, new, lm, low
        return self.__class__(lm[:self.degree]) / low[0]

    def __repr__(self):
        return repr(self.coeffs)

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        for c1, c2 in zip(self.coeffs, other.coeffs):
            if c1 != c2:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self.__class__([-c for c in self.coeffs])

    @classmethod
    def one(cls):
        return cls([1] + [0] * (cls.degree - 1))

    @classmethod
    def zero(cls):
        return cls([0] * cls.degree)

# The quadratic extension field
class FQ2(FQP):
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.modulus_coeffs = [1, 0]
        self.mc_tuples = [(0, 1)]
        self.degree = 2
        self.__class__.degree = 2

x = FQ2([1, 0])
f = FQ2([1, 2])
fpx = FQ2([2, 2])
one = FQ2.one()

# Check that the field works fine
assert x + f == fpx
assert f / f == one
assert one / f + x / f == (one + x) / f
assert one * f + x * f == (one + x) * f
assert x ** (field_modulus ** 2 - 1) == one

# The 12th-degree extension field
class FQ12(FQP):
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.modulus_coeffs = FQ12_modulus_coeffs
        self.mc_tuples = FQ12_mc_tuples
        self.degree = 12
        self.__class__.degree = 12
