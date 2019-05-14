import binary_fft as b
f = b.BinaryField(65579)
from hashlib import sha256
def hash(x): return sha256(x).digest()

log2 = b.log2

def eval_polynomial_at(poly, x):
    return f.eval_poly_at(poly, x)

def interpolate(xs, values):
    if xs == list(range(len(xs))):
        return b.invfft(f, xs, values)
    else:
        return b.interpolate(f, xs, values)

def next_power_of_two(n):
    return 2**log2(n-1) * 2

def multi_evaluate(xs, poly):
    extended_values = b.fft(f, list(range(next_power_of_two(max(xs)+1))), poly)
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
    return multi_evaluate(list(range(length)), poly)

def fill_axis(xs: List[int], values: List[Bytes32], length: int) -> List[Bytes32]:
    """
    Interprets a series of 32-byte chunks as a series of ordered packages of field
    elements. For each i, treats the set of i'th field elements in each chunk as
    evaluations of a polynomial. Evaluates the polynomials on the extended domain
    range(0, length) and provides the 32-byte chunks that are the packages of the
    extended evaluations of every polynomial at each coordinate.
    """
    data = [[bytes_to_int(a[i: FIELD_ELEMENT_BITS//8]) for a in values] for i in range(0, 32, FIELD_ELEMENT_BITS)]
    newdata = [fill(xs, d, length) for d in data]
    return [b''.join([int_to_bytes(n[i], FIELD_ELEMENT_BITS//8) for n in newdata]) for i in range(length)]

def get_data_square(data: bytes) -> List[List[Bytes32]]:
    """
    Converts data into a 2**k x 2**k square, padding with zeroes if necessary.
    """
    print("Data length:", len(data))
    chunks = [data[i: i+32] for i in range(0, len(data), 32)]
    print("Chunks:", len(chunks))
    target_size = 4**(log2(len(chunks) - 1) // 2 + 1)
    chunks.extend([ZERO_HASH for i in range(len(chunks), target_size)])
    print("Extended to:", len(chunks))
    side_length = integer_squareroot(len(chunks))
    print("Side length:", side_length)
    return [chunks[i: i + side_length] for i in range(0, len(chunks), side_length)]

def extend_data_square(data: List[List[Bytes32]]) -> List[List[Bytes32]]:
    """
    Extends a 2**k x 2**k square to 2**(k+1) x 2**(k+1) using `fill_axis` to
    fill rows and columns.
    """
    L = len(data)
    # Extend each row
    data = [fill_axis(list(range(L)), row, L * 2) for row in data]
    # Flip rows and columns
    data = [[data[j][i] for i in range(L)] for j in range(len(data))]
    # Extend each column
    data = [fill_axis(list(range(L)), row, L * 2) for row in data]
    # Flip back to row form
    data = [[data[j][i] for i in range(L)] for j in range(len(data))]
    return data

def mk_data_root(data: bytes) -> Bytes32:
    """
    Computes the root of the package of rows and colums of a given piece of data.
    """
    square = extend_data_square(get_data_square(data))
    row_roots = [get_merkle_root(r) for r in square]
    transposed_square = [[square[j][i] for i in range(len(square))] for j in range(len(square))]
    column_roots = [get_merkle_root(r) for r in transposed_square]
    return hash(get_merkle_root(row_roots) + get_merkle_root(column_roots))
