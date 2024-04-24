cache = [[None for _ in range(256)] for _ in range(256)]

def binmul(v1, v2, L=None):
    if L is None:
        L = 1 << (max(v1, v2).bit_length() - 1).bit_length()
    if v1 < 2 or v2 < 2:
        # print('base case:', v1, '*', v2, '| L', L, 'output', v1 * v2)
        return v1 * v2
    if v1 < 256 and v2 < 256 and cache[v1][v2] is not None:
        return cache[v1][v2]
    halflen = L//2
    quarterlen = L//4
    halfmask = (1 << halflen)-1

    L1, R1 = v1 & halfmask, v1 >> halflen
    L2, R2 = v2 & halfmask, v2 >> halflen
    # print('v1:', L1, R1, 'v2:', L2, R2, 'halflen:', halflen, 'halfmask:', halfmask)

    R1R2 = binmul(R1, R2, halflen)
    R1R2_high = binmul(1 << quarterlen, R1R2, halflen)
    return (
        binmul(L1, L2, halflen) ^
        R1R2 ^
        (binmul(L1, R2, halflen) << halflen) ^
        (binmul(R1, L2, halflen) << halflen) ^
        (R1R2_high << halflen)
    )

for i in range(256):
    for j in range(256):
        cache[i][j] = binmul(i, j)

class BinaryFieldElement():

    def __new__(cls, value):
        if isinstance(value, list):
            return [cls(v) for v in value]
        if isinstance(value, cls):
            return value
        instance = super(BinaryFieldElement, cls).__new__(cls)
        return instance

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        if isinstance(other, int):
            other = BinaryFieldElement(other)
        return BinaryFieldElement(self.value ^ other.value)
    
    __sub__ = __add__

    def __neg__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, int):
            other = BinaryFieldElement(other)
        return BinaryFieldElement(binmul(self.value, other.value))

    def __pow__(self, other):
        if other == 0:
            return BinaryFieldElement(1)
        elif other == 1:
            return self
        elif other == 2:
            return self * self
        else:
            return self.__pow__(other % 2) * self.__pow__(other // 2) ** 2

    def inv(self):
        L = 1 << (self.value.bit_length() - 1).bit_length()
        return self ** (2**L - 2)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = BinaryFieldElement(other)
        return BinaryFieldElement(binmul(self.value, other.inv().value))

    def __eq__(self, other):
        if isinstance(other, int):
            other = BinaryFieldElement(other)
        return self.value == other.value

    def __repr__(self):
        return '<'+str(self.value)+'>'

    def to_bytes(self, length, byteorder):
        return self.value.to_bytes(length, byteorder)

    @classmethod
    def from_bytes(cls, b, byteorder):
        return cls(int.from_bytes(b, byteorder))
