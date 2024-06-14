class FieldElement(): 
    def __init__(self, value):
        if isinstance(value, self.__class__):
            value = value.value
        self.value = value % self.modulus

    def __add__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        return self.__class__((self.value + othervalue) % self.modulus)

    def __sub__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        return self.__class__((self.value - othervalue) % self.modulus)

    def __neg__(self):
        return self.__class__(self.modulus - (self.value or self.modulus))

    def __mul__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        return self.__class__((self.value * othervalue) % self.modulus)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, other):
        return self.__class__(pow(self.value, other, self.modulus))

    def inv(self):
        return self.__class__(
            pow(self.value, -1, self.modulus) if self.value else 0
        )

    def sqrt(self):
        assert self.modulus % 4 == 3
        return self ** ((self.modulus + 1) // 4)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = self.__class__(other)
        return self * other.inv()

    def __eq__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        return self.value == othervalue

    def __repr__(self):
        return '<'+str(self.value)+'>'

    def to_bytes(self):
        return self.value.to_bytes(4, 'little')

    @classmethod
    def from_bytes(cls, bytez):
        return cls(int.from_bytes(bytez, 'little'))

class ExtendedFieldElement():
    def __init__(self, value):
        self.value = self._to_list(value)
        self.modulus = self.value[0].modulus

    def _to_list(self, value):
        if isinstance(value, self.__class__):
            return value.value
        elif isinstance(value, self.subclass):
            return [value] + [self.subclass(0)]*3
        elif isinstance(value, list):
            return [self.subclass(v) for v in value]
        elif isinstance(value, int):
            return [self.subclass(value)] + [self.subclass(0)]*3
        else:
            raise Exception("Incompatible value: {}".format(value))

    def __add__(self, other):
        othervalue = self._to_list(other)
        return self.__class__([x+y for x,y in zip(self.value, othervalue)])

    def __sub__(self, other):
        othervalue = self._to_list(other)
        return self.__class__([x-y for x,y in zip(self.value, othervalue)])

    def __mul__(self, other):
        if isinstance(other, (int, self.subclass)):
            return self.__class__([x*other for x in self.value])
        m1, m2, m3, m4 = self.value
        o1, o2, o3, o4 = self._to_list(other)
        o_LL = [m1*o1 - m2*o2, m1*o2 + m2*o1]
        o_LR = [m1*o3 - m2*o4, m1*o4 + m2*o3]
        o_RL = [m3*o1 - m4*o2, m3*o2 + m4*o1]
        o_RR = [m3*o3 - m4*o4, m3*o4 + m4*o3]
        o = [
            o_LL[0] - (o_RR[0] - o_RR[1]*self.extension_i),
            o_LL[1] - (o_RR[1] + o_RR[0]*self.extension_i),
            o_LR[0] + o_RL[0],
            o_LR[1] + o_RL[1]
        ]
        return self.__class__(o)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, other):
        if other == 0:
            return self.__class__([1,0,0,0])
        elif other == 1:
            return self
        elif other == 2:
            return self * self
        else:
            return self.__pow__(other % 2) * self.__pow__(other // 2) ** 2

    def inv(self):
        # return self ** (self.modulus ** 4 - 2)
        x0, x1, x2, x3 = self.value
        r20 = x2*x2 - x3*x3
        r21 = 2 * x2 * x3
        denom0 = x0**2 - x1**2 + r20 - r21 * 2
        denom1 = 2*x0*x1 + r21 + r20 * 2
        inv_denom_norm = (denom0 ** 2 + denom1 ** 2).inv()
        inv_denom0 = denom0 * inv_denom_norm
        inv_denom1 = -denom1 * inv_denom_norm
        o = self.__class__([
            x0 * inv_denom0 - x1 * inv_denom1,
            x0 * inv_denom1 + x1 * inv_denom0,
            -x2 * inv_denom0 + x3 * inv_denom1,
            -x2 * inv_denom1 - x3 * inv_denom0,
        ])
        return o
    
    def __truediv__(self, other):
        other = self.__class__(self._to_list(other))
        if other.value[1:] == [0,0,0]:
            factor = other.value[0].inv()
            return self.__class__([v * factor for v in self.value])
        else:
            return self * other.inv()

    def __eq__(self, other):
        return self.value == self._to_list(other)

    def __repr__(self):
        return '<'+str([v.value for v in self.value])+'>'

    def to_bytes(self):
        return b''.join([v.to_bytes() for v in self.value])

    @classmethod
    def from_bytes(cls, bytez):
        return cls([
            int.from_bytes(bytez[i:i+4], 'little') for i in range(0, 16, 4)
        ])

class SmallMersenneElement(FieldElement):
    modulus = 2**5-1

class MediumMersenneElement(FieldElement):
    modulus = 2**17-1

class BigMersenneElement(FieldElement):
    modulus = 2**31-1

class ExtendedSmallMersenneElement(ExtendedFieldElement):
    subclass = SmallMersenneElement
    extension_i = 4

class ExtendedMediumMersenneElement(ExtendedFieldElement):
    subclass = MediumMersenneElement
    extension_i = 3

class ExtendedBigMersenneElement(ExtendedFieldElement):
    subclass = BigMersenneElement
    extension_i = 2

S = SmallMersenneElement
M = MediumMersenneElement
B = BigMersenneElement
ES = ExtendedSmallMersenneElement
EM = ExtendedMediumMersenneElement
EB = ExtendedBigMersenneElement
