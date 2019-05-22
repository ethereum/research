# import fast_binary_fft as b
import binary_fft as b
f = b.BinaryField(65579)
from hashlib import sha256
def hash(x): return sha256(x).digest()

log2 = b.log2

# Alternative optimized/simplified inverse-FFT implementation

# shift_polys[i][j] is the 2**j degree coefficient of the polynomial that evaluates to [1,1...1, 0,0....0] with 2**(i-1) ones and 2**(i-1) zeroes
shift_polys = [[], [1], [32755, 32755], [52774, 60631, 8945], [38902, 5560, 44524, 12194], [55266, 46488, 60321, 5401, 40130], [21827, 32224, 51565, 15072, 8277, 64379], [59460, 15452, 60370, 24737, 20321, 35516, 39606], [42623, 56997, 25925, 15351, 16625, 47045, 38250, 17462], [7575, 27410, 32434, 22187, 28933, 15447, 37964, 38186, 4776], [39976, 61188, 42456, 2155, 6178, 34033, 52305, 14913, 2896, 48908], [6990, 12021, 36054, 16198, 17011, 14018, 58553, 13272, 25318, 5288, 21429], [16440, 34925, 14360, 22561, 43883, 36645, 7613, 26531, 8597, 59502, 61283, 53412]]

def invfft2(field, vals):
    if len(vals) == 1:
        return [vals[0]]
    L = invfft2(field, vals[:len(vals)//2])
    R = b.shift(field, invfft2(field, vals[len(vals)//2:]), len(vals)//2)
    o = [0] * len(vals)
    for j, (l, r) in enumerate(zip(L, R)):
        o[j] ^= l
        for i, coeff in enumerate(shift_polys[log2(len(vals))]):
            o[2**i+j] ^= field.mul(l ^ r, coeff)
    # print(vals, o)
    return o

# Alternative simplified but less efficient FFT implementation
def p_mod_shift(field, poly):
    shift_poly = shift_polys[log2(len(poly))]
    half_height = len(poly)//2
    low = poly[::]
    high = []
    while len(high) < half_height:
        high.insert(0, field.div(low[-1], shift_poly[-1]))
        for i, coeff in enumerate(shift_poly[:-1]):
            low[-half_height-1+2**i] ^= field.mul(high[0], coeff)
        low.pop()
    return high, low

def fft2(field, poly):
    # if len(poly) == 1:
    #     return [poly[0]]
    if len(poly) <= 8:
        return b._simple_ft(field, list(range(len(poly))), poly)
    # p(x) = high(x) * s(x) + low(x)
    high, low = p_mod_shift(field, poly)
    # p(x) = g(x) * (s(x)+1) + h(x) * s(x)
    g, h = low, [x^y for x,y in zip(high, low)]
    # print(poly, g, h)
    g_values = fft2(field, g)
    h_values = fft2(field, b.shift(field, h, len(poly)//2))
    return g_values + h_values

def eval_polynomial_at(poly, x):
    return f.eval_poly_at(poly, x)

def interpolate(xs, values):
    if xs == list(range(len(xs))):
        return invfft2(f, values)
    else:
        return b.interpolate(f, xs, values)

def next_power_of_two(n):
    return 2**log2(n-1) * 2

def multi_evaluate(xs, poly):
    # degree = next_power_of_two(max(xs)+1)
    extended_values = b.fft(f, list(range(next_power_of_two(max(xs)+1))), poly)
    # extended_values = fft2(f, poly + [0] * (degree - len(poly)))
    return [extended_values[x] for x in xs]

def get_merkle_root(values):
    extended_length = 2**log2(len(values))
    tree = [None] * extended_length + values + [b'\x00'*32] * (extended_length - len(values))
    for i in range(extended_length - 1, 0, -1):
        tree[i] = hash(tree[i*2] + tree[i*2+1])
    return tree[1]

def int_to_bytes(i, n):
    return i.to_bytes(n, 'little')

def bytes_to_int(b):
    return int.from_bytes(b, 'little')

def integer_squareroot(n: int) -> int:
    """
    The largest integer ``x`` such that ``x**2`` is less than or equal to ``n``.
    """
    assert n >= 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

Bytes32 = 0
List = {int: 1, Bytes32: 2, 2: 2}
FIELD_ELEMENT_BITS = 16
ZERO_HASH = b'\x00' * 32

def fill(xs: List[int], values: List[int], length: int) -> List[int]:
    """
    Takes the minimal polynomial that returns values[i] at xs[i] and computes
    its outputs for all values in range(0, length)
    """
    poly = interpolate(xs, values)
    # o = values + multi_evaluate(list(range(len(values))), b.shift(f, poly, len(values)))
    # return o
    return multi_evaluate(list(range(length)), poly)

def fill_axis(xs: List[int], values: List[Bytes32], length: int) -> List[Bytes32]:
    """
    Interprets a series of 32-byte chunks as a series of ordered packages of field
    elements. For each i, treats the set of i'th field elements in each chunk as
    evaluations of a polynomial. Evaluates the polynomials on the extended domain
    range(0, length) and provides the 32-byte chunks that are the packages of the
    extended evaluations of every polynomial at each coordinate.
    """
    data = [[bytes_to_int(a[i: i + FIELD_ELEMENT_BITS//8]) for a in values] for i in range(0, 32, FIELD_ELEMENT_BITS//8)]
    newdata = [fill(xs, d, length) for d in data]
    return [b''.join([int_to_bytes(n[i], FIELD_ELEMENT_BITS//8) for n in newdata]) for i in range(length)]

def get_data_square(data: bytes) -> List[List[Bytes32]]:
    """
    Converts data into a 2**k x 2**k square, padding with zeroes if necessary.
    """
    print("Data length:", len(data))
    chunks = [data[i: i+32] for i in range(0, len(data), 32)]
    if len(chunks[-1]) < 32:
        chunks[-1] += b'\x00' * (32 - len(chunks[-1]))
    print("Chunks:", len(chunks))
    target_size = 4**(log2(len(chunks) - 1) // 2 + 1)
    chunks.extend([ZERO_HASH for i in range(len(chunks), target_size)])
    print("Extended to:", len(chunks))
    side_length = integer_squareroot(len(chunks))
    print("Side length:", side_length)
    return [chunks[i: i + side_length] for i in range(0, len(chunks), side_length)]

def extend_data_square(square: List[List[Bytes32]]) -> List[List[Bytes32]]:
    """
    Extends a 2**k x 2**k square to 2**(k+1) x 2**(k+1) using `fill_axis` to
    fill rows and columns.
    """
    L = len(square)
    # Extend each row
    square = [fill_axis(list(range(L)), row, L * 2) for row in square]
    # Flip rows and columns
    square = [[square[j][i] for i in range(L)] for j in range(len(square))]
    # Extend each column
    square = [fill_axis(list(range(L)), row, L * 2) for row in square]
    # Flip back to row form
    square = [[square[j][i] for i in range(L)] for j in range(len(square))]
    return square

def mk_data_root(data: bytes) -> Bytes32:
    """
    Computes the root of the package of rows and colums of a given piece of data.
    """
    square = extend_data_square(get_data_square(data))
    row_roots = [get_merkle_root(r) for r in square]
    transposed_square = [[square[j][i] for i in range(len(square))] for j in range(len(square))]
    column_roots = [get_merkle_root(r) for r in transposed_square]
    return hash(get_merkle_root(row_roots) + get_merkle_root(column_roots))
