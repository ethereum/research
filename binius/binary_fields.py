from hashlib import sha256

def hash(x):
    return sha256(x).digest()

def binlength(v):
    return 1 << (v.bit_length() - 1).bit_length()

def binmul(v1, v2, L=None):
    if L is None:
        L = binlength(max(v1, v2))
    if v1 < 2 or v2 < 2:
        # print('base case:', v1, '*', v2, '| L', L, 'output', v1 * v2)
        return v1 * v2
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

    def to_bytes(self, v, byteorder):
        return self.value.to_bytes(v, byteorder)

    @classmethod
    def from_bytes(cls, b, byteorder):
        return cls(int.from_bytes(b, byteorder))

def get_class(arg, start=int):
    if isinstance(arg, (list, tuple)):
        output = start
        for a in arg:
            output = get_class(a, output)
        return output
    elif start == int:
        return arg.__class__
    elif arg.__class__ == int:
        return start
    elif start == arg.__class__:
        return arg.__class__
    else:
        raise Exception("Incompatible classes: {} {}".format(start, arg.__class__))

def eval_poly_at(poly, pt):
    cls = get_class([poly, pt])
    o = cls(0)
    power = cls(1)
    for coeff in poly:
        o += coeff * power
        power *= pt
    return o

def mul_polys(a, b):
    cls = get_class([a,b])
    o = [cls(0)] * (len(a) + len(b) - 1)
    for i, aval in enumerate(a):
        for j, bval in enumerate(b):
            o[i+j] += a[i] * b[j]
    return o

def compute_lagrange_poly(size, pt):
    cls = get_class(pt)
    opoly = [cls(1)]
    ofactor = cls(1)
    for i in range(size):
        _i = cls(i)
        if _i != pt:
            opoly = mul_polys(opoly, [-_i, 1])
            ofactor *= (pt - _i)
    return [x/ofactor for x in opoly]

def multilinear_poly_eval(evals, pt):
    cls = get_class([evals, pt])
    assert len(evals) == 2 ** len(pt)
    o = cls(0)
    for i, evaluation in enumerate(evals):
        value = evals[i]
        for j, coord in enumerate(pt):
            if (i >> j) % 2:
                value *= coord
            else:
                value *= (cls(1) - coord)
        o += value
    return o

def extend(vals, expansion_factor=2):
    cls = get_class(vals)
    lagranges = [
        compute_lagrange_poly(len(vals), cls(i))
        for i in range(len(vals))
    ]
    output = vals[::]
    for x in range(len(vals), expansion_factor * len(vals)):
        _x = cls(x)
        o = cls(0)
        for v, L in zip(vals, lagranges):
            o += eval_poly_at(L, x) * v
        output.append(o)
    return output

def merkelize(vals):
    assert len(vals) & (len(vals)-1) == 0
    o = [None] * len(vals) + [hash(x) for x in vals]
    for i in range(len(vals)-1, 0, -1):
        o[i] = hash(o[i*2] + o[i*2+1])
    return o

def get_root(tree):
    return tree[1]

def get_branch(tree, pos):
    offset_pos = pos + len(tree) // 2
    branch_length = len(tree).bit_length() - 2
    return [tree[(offset_pos >> i)^1] for i in range(branch_length)]

def verify_branch(root, pos, val, branch):
    x = hash(val)
    for b in branch:
        if pos & 1:
            x = hash(b + x)
        else:
            x = hash(x + b)
        pos //= 2
    return x == root

EXPANSION_FACTOR = 8
NUM_CHALLENGES = 4

def evaluation_tensor_product(pt):
    cls = get_class(pt)
    o = [cls(1)]
    for coord in pt:
        o = [
            (cls(1) - coord) * v for v in o
        ] + [
            coord * v for v in o
        ]
    return o

def simple_binius_proof(evals, evaluation_point):
    cls = get_class([evals, evaluation_point])
    L = len(evals).bit_length() - 1
    row_length = 1 << (L // 2)
    row_count = 1 << ((L + 1) // 2)
    rows = [evals[i:i+row_length] for i in range(0, len(evals), row_length)]
    extended_rows = [extend(row, expansion_factor=EXPANSION_FACTOR) for row in rows]
    extended_row_length = row_length * EXPANSION_FACTOR
    row_combination = evaluation_tensor_product(evaluation_point[L//2:])

    assert len(row_combination) == len(rows) == row_count
    t_prime = [
        sum([rows[i][j] * row_combination[i] for i in range(row_count)], cls(0))
        for j in range(row_length)
    ]
    columns = [[row[j] for row in extended_rows] for j in range(extended_row_length)]
    bytes_per_element = 1
    while 256**bytes_per_element < row_length:
        bytes_per_element *= 2
    packed_columns = [b''.join(x.to_bytes(bytes_per_element, 'little') for x in col) for col in columns]
    merkle_tree = merkelize(packed_columns)
    root = get_root(merkle_tree)
    challenges = [
        int.from_bytes(hash(root + i.to_bytes(4, 'little')), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]
    return {
        'root': root,
        'evaluation_point': evaluation_point,
        'eval': multilinear_poly_eval(evals, evaluation_point),
        't_prime': t_prime,
        'columns': [columns[c] for c in challenges],
        'branches': [get_branch(merkle_tree, c) for c in challenges],
    }

def verify_simple_binius_proof(proof):
    cls = get_class([proof['columns'], proof['evaluation_point'], proof['eval']])
    # Check Merkle branches
    root = proof["root"]
    extended_row_length = 2**len(proof["branches"][0])
    challenges = [
        int.from_bytes(hash(root + i.to_bytes(4, 'little')), 'little') % extended_row_length
        for i in range(NUM_CHALLENGES)
    ]
    row_length = extended_row_length // EXPANSION_FACTOR
    row_count = len(proof['columns'][0])
    bytes_per_element = 1
    while 256**bytes_per_element < row_length:
        bytes_per_element *= 2
    for challenge, branch, column in zip(challenges, proof['branches'], proof['columns']):
        packed_column = b''.join(x.to_bytes(bytes_per_element, 'little') for x in column)
        print(f"Verifying Merkle branch for column {column}")
        assert verify_branch(root, challenge, packed_column, branch)
    extended_t_prime = extend(proof["t_prime"], expansion_factor=EXPANSION_FACTOR)
    log_row_count = row_count.bit_length() - 1
    row_combination = evaluation_tensor_product(proof["evaluation_point"][log_row_count:])
    for column, challenge in zip(proof['columns'], challenges):
        expected_tprime = sum([column[i] * row_combination[i] for i in range(row_count)], cls(0))
        print(f"Testing challenge on column {challenge}: expected {expected_tprime} computed {extended_t_prime[challenge]}")
        assert expected_tprime == extended_t_prime[challenge]
    col_combination = evaluation_tensor_product(proof["evaluation_point"][:log_row_count])
    computed_eval = sum([proof["t_prime"][i] * col_combination[i] for i in range(row_length)], cls(0))
    print(f"Testing evaluation: expected {proof['eval']} computed {computed_eval}")
    assert computed_eval == proof["eval"]
    return True
