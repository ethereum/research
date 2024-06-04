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
        if other == 0:
            return self.__class__(1)
        elif other == 1:
            return self
        elif other == 2:
            return self * self
        else:
            return self.__pow__(other % 2) * self.__pow__(other // 2) ** 2

    def inv(self):
        return self ** (self.modulus - 2)

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

class SmallMersenneElement(FieldElement):
    modulus = 2**5-1

class MediumMersenneElement(FieldElement):
    modulus = 2**17-1

class BigMersenneElement(FieldElement):
    modulus = 2**31-1
