import copy


# Galois field class and logtable
#
# See: https://en.wikipedia.org/wiki/Finite_field
#
# Note that you can substitute "Galois" with "float" in the code, and
# the code will then magically start using the plain old field of rationals
# instead of this spooky modulo polynomial thing. If you are not an expert in
# finite field theory and want to dig deep into how this code works, I
# recommend adding the line "Galois = float" immediately after this class (and
# not using the methods that require serialization)
#
# As a quick intro to finite field theory, the idea is that there exist these
# things called fields, which are basically sets of objects together with
# rules for addition, subtraction, multiplication, division, such that algebra
# within this field is consistent, even if the results look nonsensical from
# a "normal numbers" perspective. For instance, consider the field of integers
# modulo 7. Here, for example, 2 * 5 = 3, 3 * 4 = 5, 6 * 6 = 1, 6 + 6 = 5.
# However, all algebra still works; for example, (a^2 - b^2) = (a + b)(a - b)
# works for all a,b. For this reason, we can do secret sharing arithmetic
# "over" any field. The reason why Galois fields are preferable is that all
# elements in the Galois field are values in [0 ... 255] (at least using the
# canonical serialization that we use here); no amount of addition,
# multiplication, subtraction or division will ever get you anything else.
# This guarantees that our secret shares will always be serializable as byte
# arrays. The way the Galois field we use here works is that the elements are
# polynomials of elements in the field of integers mod 2, so addition and
# subtraction are xor, and multiplication is modulo x^8 + x^4 + x^3 + x + 1,
# and division is defined by a/b = c iff bc = a and b != 0. In practice, we
# do multiplication and division via a precomputed log table using x+1 as a
# base

# per-byte 2^8 Galois field
# Note that this imposes a hard limit that the number of extended chunks can
# be at most 256 along each dimension


def galoistpl(a):
    # 2 is not a primitive root, so we have to use 3 as our logarithm base
    unrolla = [a/(2**k) % 2 for k in range(8)]
    res = [0] + unrolla
    for i in range(8):
        res[i] = (res[i] + unrolla[i]) % 2
    if res[-1] == 0:
        res.pop()
    else:
        # AES Polynomial
        for i in range(9):
            res[i] = (res[i] - [1, 1, 0, 1, 1, 0, 0, 0, 1][i]) % 2
        res.pop()
    return sum([res[k] * 2**k for k in range(8)])

# Precomputing a multiplication and XOR table for increased speed
glogtable = [0] * 256
gexptable = []
v = 1
for i in range(255):
    glogtable[v] = i
    gexptable.append(v)
    v = galoistpl(v)


class Galois:
    val = 0

    def __init__(self, val):
        self.val = val.val if isinstance(self.val, Galois) else val

    def __add__(self, other):
        return Galois(self.val ^ other.val)

    def __mul__(self, other):
        if self.val == 0 or other.val == 0:
            return Galois(0)
        return Galois(gexptable[(glogtable[self.val] +
                                 glogtable[other.val]) % 255])

    def __sub__(self, other):
        return Galois(self.val ^ other.val)

    def __div__(self, other):
        if other.val == 0:
            raise ZeroDivisionError
        if self.val == 0:
            return Galois(0)
        return Galois(gexptable[(glogtable[self.val] -
                                 glogtable[other.val]) % 255])

    def __int__(self):
        return self.val

    def __repr__(self):
        return repr(self.val)

# Evaluates a polynomial in little-endian form, eg. x^2 + 3x + 2 = [2, 3, 1]
# (normally I hate little-endian, but in this case dealing with polynomials
# it's justified, since you get the nice property that p[n] is the nth degree
# term of p) at coordinate x, eg. eval_poly_at([2, 3, 1], 5) = 42 if you are
# using float as your arithmetic


def eval_poly_at(p, x):
    arithmetic = p[0].__class__
    y = arithmetic(0)
    x_to_the_i = arithmetic(1)
    for i in range(len(p)):
        y += x_to_the_i * p[i]
        x_to_the_i *= x
    return y


# Given p+1 y values and x values with no errors, recovers the original
# p+1 degree polynomial. For example,
# lagrange_interp([51.0, 59.0, 66.0], [1, 3, 4]) = [50.0, 0, 1.0]
# if you are using float as your arithmetic


def lagrange_interp(pieces, xs):
    arithmetic = pieces[0].__class__
    zero, one = arithmetic(0), arithmetic(1)
    # Generate master numerator polynomial
    root = [one]
    for i in range(len(xs)):
        root.insert(0, zero)
        for j in range(len(root)-1):
            root[j] = root[j] - root[j+1] * xs[i]
    # Generate per-value numerator polynomials by dividing the master
    # polynomial back by each x coordinate
    nums = []
    for i in range(len(xs)):
        output = []
        last = one
        for j in range(2, len(root)+1):
            output.insert(0, last)
            if j != len(root):
                last = root[-j] + last * xs[i]
        nums.append(output)
    # Generate denominators by evaluating numerator polys at their x
    denoms = []
    for i in range(len(xs)):
        denom = zero
        x_to_the_j = one
        for j in range(len(nums[i])):
            denom += x_to_the_j * nums[i][j]
            x_to_the_j *= xs[i]
        denoms.append(denom)
    # Generate output polynomial
    b = [zero for i in range(len(pieces))]
    for i in range(len(xs)):
        yslice = pieces[int(i)] / denoms[int(i)]
        for j in range(len(pieces)):
            b[j] += nums[i][j] * yslice
    return b


# Compresses two linear equations of length n into one
# equation of length n-1
# Format:
# 3x + 4y = 80 (ie. 3x + 4y - 80 = 0) -> a = [3,4,-80]
# 5x + 2y = 70 (ie. 5x + 2y - 70 = 0) -> b = [5,2,-70]


def elim(a, b):
    aprime = [x*b[0] for x in a]
    bprime = [x*a[0] for x in b]
    c = [aprime[i] - bprime[i] for i in range(1, len(a))]
    return c


# Linear equation solver
# Format:
# 3x + 4y = 80, y = 5 (ie. 3x + 4y - 80z = 0, y = 5, z = 1)
#      -> coeffs = [3,4,-80], vals = [5,1]


def evaluate(coeffs, vals):
    arithmetic = coeffs[0].__class__
    tot = arithmetic(0)
    for i in range(len(vals)):
        tot -= coeffs[i+1] * vals[i]
    if int(coeffs[0]) == 0:
        raise ZeroDivisionError
    return tot / coeffs[0]


# Linear equation system solver
# Format:
# ax + by + c = 0, dx + ey + f = 0
# -> [[a, b, c], [d, e, f]]
# eg.
# [[3.0, 5.0, -13.0], [9.0, 1.0, -11.0]] -> [1.0, 2.0]


def sys_solve(eqs):
    arithmetic = eqs[0][0].__class__
    one = arithmetic(1)
    back_eqs = [eqs[0]]
    while len(eqs) > 1:
        neweqs = []
        for i in range(len(eqs)-1):
            neweqs.append(elim(eqs[i], eqs[i+1]))
        eqs = neweqs
        i = 0
        while i < len(eqs) - 1 and int(eqs[i][0]) == 0:
            i += 1
        back_eqs.insert(0, eqs[i])
    kvals = [one]
    for i in range(len(back_eqs)):
        kvals.insert(0, evaluate(back_eqs[i], kvals))
    return kvals[:-1]


def polydiv(Q, E):
    qpoly = copy.deepcopy(Q)
    epoly = copy.deepcopy(E)
    div = []
    while len(qpoly) >= len(epoly):
        div.insert(0, qpoly[-1] / epoly[-1])
        for i in range(2, len(epoly)+1):
            qpoly[-i] -= div[0] * epoly[-i]
        qpoly.pop()
    return div


# Given a set of y coordinates and x coordinates, and the degree of the
# original polynomial, determines the original polynomial even if some of
# the y coordinates are wrong. If m is the minimal number of pieces (ie.
# degree + 1), t is the total number of pieces provided, then the algo can
# handle up to (t-m)/2 errors. See:
# http://en.wikipedia.org/wiki/Berlekamp%E2%80%93Welch_algorithm#Example
# (just skip to my example, the rest of the article sucks imo)


def berlekamp_welch_attempt(pieces, xs, master_degree):
    error_locator_degree = (len(pieces) - master_degree - 1) / 2
    arithmetic = pieces[0].__class__
    zero, one = arithmetic(0), arithmetic(1)
    # Set up the equations for y[i]E(x[i]) = Q(x[i])
    # degree(E) = error_locator_degree
    # degree(Q) = master_degree + error_locator_degree - 1
    eqs = []
    for i in range(2 * error_locator_degree + master_degree + 1):
        eqs.append([])
    for i in range(2 * error_locator_degree + master_degree + 1):
        neg_x_to_the_j = zero - one
        for j in range(error_locator_degree + master_degree + 1):
            eqs[i].append(neg_x_to_the_j)
            neg_x_to_the_j *= xs[i]
        x_to_the_j = one
        for j in range(error_locator_degree + 1):
            eqs[i].append(x_to_the_j * pieces[i])
            x_to_the_j *= xs[i]
    # Solve 'em
    # Assume the top error polynomial term to be one
    errors = error_locator_degree
    ones = 1
    while errors >= 0:
        try:
            polys = sys_solve(eqs) + [one] * ones
            qpoly = polys[:errors + master_degree + 1]
            epoly = polys[errors + master_degree + 1:]
            break
        except ZeroDivisionError:
            for eq in eqs:
                eq[-2] += eq[-1]
                eq.pop()
            eqs.pop()
            errors -= 1
            ones += 1
    if errors < 0:
        raise Exception("Not enough data!")
    # Divide the polynomials
    qpoly = polys[:error_locator_degree + master_degree + 1]
    epoly = polys[error_locator_degree + master_degree + 1:]
    div = []
    while len(qpoly) >= len(epoly):
        div.insert(0, qpoly[-1] / epoly[-1])
        for i in range(2, len(epoly)+1):
            qpoly[-i] -= div[0] * epoly[-i]
        qpoly.pop()
    # Check
    corrects = 0
    for i, x in enumerate(xs):
        if int(eval_poly_at(div, x)) == int(pieces[i]):
            corrects += 1
    if corrects < master_degree + errors:
        raise Exception("Answer doesn't match (too many errors)!")
    return div


# Extends a list of integers in [0 ... 255] (if using Galois arithmetic) by
# adding n redundant error-correction values


def extend(data, n, arithmetic=Galois):
    data2 = map(arithmetic, data)
    data3 = data[:]
    poly = berlekamp_welch_attempt(data2,
                                   map(arithmetic, range(len(data))),
                                   len(data) - 1)
    for i in range(n):
        data3.append(int(eval_poly_at(poly, arithmetic(len(data) + i))))
    return data3


# Repairs a list of integers in [0 ... 255]. Some integers can be erroneous,
# and you can put None in place of an integer if you know that a certain
# value is defective or missing. Uses the Berlekamp-Welch algorithm to
# do error-correction


def repair(data, datasize, arithmetic=Galois):
    vs, xs = [], []
    for i, v in enumerate(data):
        if v is not None:
            vs.append(arithmetic(v))
            xs.append(arithmetic(i))
    poly = berlekamp_welch_attempt(vs, xs, datasize - 1)
    return [int(eval_poly_at(poly, arithmetic(i))) for i in range(len(data))]


# Extends a list of bytearrays
# eg. extend_chunks([map(ord, 'hello'), map(ord, 'world')], 2)
# n is the number of redundant error-correction chunks to add


def extend_chunks(data, n, arithmetic=Galois):
    o = []
    for i in range(len(data[0])):
        o.append(extend(map(lambda x: x[i], data), n, arithmetic))
    return map(list, zip(*o))


# Repairs a list of bytearrays. Use None in place of a missing array.
# Individual arrays can contain some missing or erroneous data.


def repair_chunks(data, datasize, arithmetic=Galois):
    first_nonzero = 0
    while not data[first_nonzero]:
        first_nonzero += 1
    for i in range(len(data)):
        if data[i] is None:
            data[i] = [None] * len(data[first_nonzero])
    o = []
    for i in range(len(data[0])):
        o.append(repair(map(lambda x: x[i], data), datasize, arithmetic))
    return map(list, zip(*o))


# Extends either a bytearray or a list of bytearrays or a list of lists...
# Used in the cubify method to expand a cube in all dimensions


def deep_extend_chunks(data, n, arithmetic=Galois):
    if not isinstance(data[0], list):
        return extend(data, n, arithmetic)
    else:
        o = []
        for i in range(len(data[0])):
            o.append(
                deep_extend_chunks(map(lambda x: x[i], data), n, arithmetic))
        return map(list, zip(*o))


# ISO/IEC 7816-4 padding


def pad(data, size):
    data = data[:]
    data.append(128)
    while len(data) % size != 0:
        data.append(0)
    return data


# Removes ISO/IEC 7816-4 padding


def unpad(data):
    data = data[:]
    while data[-1] != 128:
        data.pop()
    data.pop()
    return data


# Splits a bytearray into a given number of chunks with some
# redundant chunks


def split(data, numchunks, redund):
    chunksize = len(data) / numchunks + 1
    data = pad(data, chunksize)
    chunks = []
    for i in range(0, len(data), chunksize):
        chunks.append(data[i: i+chunksize])
    o = extend_chunks(chunks, redund)
    return o


# Recombines chunks into the original bytearray


def recombine(chunks, datalength):
    datasize = datalength / len(chunks[0]) + 1
    c = repair_chunks(chunks, datasize)
    return unpad(sum(c[:datasize], []))


h = '0123456789abcdef'
hexfy = lambda x: h[x//16]+h[x % 16]
unhexfy = lambda x: h.find(x[0]) * 16 + h.find(x[1])
split2 = lambda x: map(lambda a: ''.join(a), zip(x[::2], x[1::2]))


# Canonical serialization. First argument is a bytearray, remaining
# arguments are strings to prepend


def serialize_chunk(*args):
    chunk = args[0]
    if not chunk or chunk[0] is None:
        return None
    metadata = args[1:]
    return '-'.join(map(str, metadata) + [''.join(map(hexfy, chunk))])


def deserialize_chunk(chunk):
    data = chunk.split('-')
    metadata, main = data[:-1], data[-1]
    return metadata, map(unhexfy, split2(main))


# Splits a string into a given number of chunks with some redundant chunks


def split_file(f, numchunks=5, redund=5):
    f = map(ord, f)
    ec = split(f, numchunks, redund)
    o = []
    for i, c in enumerate(ec):
        o.append(
            serialize_chunk(c, *[i, numchunks, numchunks + redund, len(f)]))
    return o


def recombine_file(chunks):
    chunks2 = map(deserialize_chunk, chunks)
    metadata = map(int, chunks2[0][0])
    o = [None] * metadata[2]
    for chunk in chunks2:
        o[int(chunk[0][0])] = chunk[1]
    return ''.join(map(chr, recombine(o, metadata[3])))

outersplitn = lambda x, k: map(lambda i: x[i:i+k], range(len(x)))


# Splits a bytearray into a hypercube with `dim` dimensions with the original
# data being in a sub-cube of width `width` and the expanded cube being of
# width `width+redund`. The cube is self-healing; if any edge in any dimension
# has missing or erroneous pieces, we can use the Berlekamp-Welch algorithm
# to fix this


def cubify(f, width, dim, redund):
    chunksize = len(f) / width**dim + 1
    data = pad(f, width**dim)
    chunks = []
    for i in range(0, len(data), chunksize * width):
        for j in range(width):
            chunks.append(data[i+j*chunksize: i+j*chunksize+chunksize])

    for i in range(dim):
        o = []
        for j in range(0, len(chunks), width):
            e = chunks[j: j + width]
            o.append(
                deep_extend_chunks(e, redund))
        chunks = o

    return chunks[0]


# `pos` is an array of coordinates. Go deep into a nested list


def descend(obj, pos):
    for p in pos:
        obj = obj[p]
    return obj


# Go deep into a nested list and modify the value


def descend_and_set(obj, pos, val):
    immed = descend(obj, pos[:-1])
    immed[pos[-1]] = val


# Use the Berlekamp-Welch algorithm to try to "heal" a particular missing
# or damaged coordinate


def heal_cube(cube, width, dim, pos, datalen):
    for d in range(len(pos)):
        o = []
        for i in range(len(cube)):
            o.append(descend(cube, pos[:d] + [i] + pos[d+1:]))
        try:
            o = repair_chunks(o, width)
            for i in range(len(cube)):
                path = pos[:d] + [i] + pos[d+1:]
                descend_and_set(cube, path, o[i])
        except:
            pass


def pack_metadata(meta):
    return map(str, meta['coords']) + [
        str(meta['base_width']),
        str(meta['extended_width']),
        str(meta['filesize'])
    ]


def unpack_metadata(meta):
    return {
        'coords': map(int, meta[:-3]),
        'base_width': int(meta[-3]),
        'extended_width': int(meta[-2]),
        'filesize': int(meta[-1])
    }


# Helper to serialize the contents of a cube of byte arrays


def _ser(chunk, meta):
    if chunk is None or (not isinstance(chunk[0], list) and
                         chunk[0] is not None):
        u = serialize_chunk(chunk, *pack_metadata(meta))
        return u
    else:
        o = []
        for i, c in enumerate(chunk):
            meta2 = copy.deepcopy(meta)
            meta2['coords'] += [i]
            o.append(_ser(c, meta2))
        return o


# Converts a deep list into a shallow list


def flatten(chunks):
    if not isinstance(chunks, list):
        return [chunks]
    else:
        o = []
        for c in chunks:
            o.extend(flatten(c))
        return o


# Converts a file into a multidimensional set of chunks with
# the desired parameters


def serialize_cubify(f, width, dim, redund):
    f = map(ord, f)
    cube = cubify(f, width, dim, redund)
    metadata = {
        'base_width': width,
        'extended_width': width + redund,
        'coords': [],
        'filesize': len(f)
    }
    cube_of_serialized_chunks = _ser(cube, metadata)
    return flatten(cube_of_serialized_chunks)


# Converts a set of serialized chunks into a partially filled cube


def construct_cube(pieces):
    pieces = map(deserialize_chunk, pieces)
    metadata = unpack_metadata(pieces[0][0])
    dim = len(metadata['coords'])
    cube = None
    for i in range(dim):
        cube = [copy.deepcopy(cube) for i in range(metadata['extended_width'])]
    for p in pieces:
        descend_and_set(cube, unpack_metadata(p[0])['coords'], p[1])
    return cube


# Tries to recreate the chunk at a particular coordinate given a set of
# other chunks


def heal_set(pieces, coords):
    c = construct_cube(pieces)
    metadata, piecezzz = deserialize_chunk(pieces[0])
    metadata = unpack_metadata(metadata)
    heal_cube(c,
              metadata['base_width'],
              len(metadata['coords']),
              coords,
              metadata['filesize'])
    metadata2 = copy.deepcopy(metadata)
    metadata2["coords"] = []
    return filter(lambda x: x, flatten(_ser(c, metadata2)))


def number_to_coords(n, w, dim):
    c = [0] * dim
    for i in range(dim):
        c[i] = n / w**(dim - i - 1)
        n %= w**(dim - i - 1)
    return c


def full_heal_set(pieces):
    c = construct_cube(pieces)
    metadata, piecezzz = deserialize_chunk(pieces[0])
    metadata = unpack_metadata(metadata)
    while 1:
        done = True
        unfilled = False
        i = 0
        while i < metadata['extended_width'] ** len(metadata['coords']):
            coords = number_to_coords(i,
                                      metadata['extended_width'],
                                      len(metadata['coords']))
            v = descend(c, coords)
            heal_cube(c,
                      metadata['base_width'],
                      len(metadata['coords']),
                      coords,
                      metadata['filesize'])
            v2 = descend(c, coords)
            if v != v2:
                done = False
            if v is None and v2 is None:
                unfilled = True
            i += 1
        if done and not unfilled:
            break
        elif done and unfilled:
            raise Exception("not enough data or too much corrupted data")
    o = []
    for i in range(metadata['base_width'] ** len(metadata['coords'])):
        coords = number_to_coords(i,
                                  metadata['base_width'],
                                  len(metadata['coords']))
        o.extend(descend(c, coords))
    return ''.join(map(chr, unpad(o)))
