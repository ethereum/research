from zorch import koalabear

#M = koalabear.KoalaBear([[1 if i==j else 0 for i in range(16)] for j in range(16)])
M = 1 / koalabear.KoalaBear([[1+i+j for i in range(16)] for j in range(16)])

def matmul_layer(values):
    orig_shape = values.shape
    size = values.shape[-1] // 16
    values = values.reshape(values.shape[:-1] + (size, 16))
    values = koalabear.matmul(values, M)
    values = values.reshape(orig_shape)
    return values


def chi_weights(coords, start_weights=None):
    if start_weights is None:
        weights = koalabear.ExtendedKoalaBear([[1,0,0,0]])
    else:
        weights = start_weights
    for c in coords:
        L = weights * (1 - c)
        R = weights * c
        weights = koalabear.ExtendedKoalaBear.append(L, R)
    return weights

def mle_eval(cube, coords):
    for coord in coords:
        b = cube[::2]
        m = cube[1::2] - b
        cube = b + m * coord
    return cube[0]

def chi_eval(source_coords, eval_coords):
    o = 1
    for s, e in zip(source_coords, eval_coords):
        o *= (s * e + (1-s) * (1-e))
    return o

def compute_lower_order_weights(lower_coords):
    return koalabear.matmul(M.swapaxes(0,1), chi_weights(lower_coords))

def compute_weights(coords):
    intermediate = compute_lower_order_weights(coords[:4])
    return chi_weights(coords[4:], start_weights=intermediate)

def fast_point_eval(source_coords, eval_coords):
    intermediate = compute_lower_order_weights(source_coords[:4])
    o = mle_eval(intermediate, eval_coords[:4])
    o *= chi_eval(source_coords[4:], eval_coords[4:])
    return o

def log2(x):
    return 0 if x <= 1 else 1 + log2(x//2)

def generate_weights_seed_coords(randomness, count, hash):
    return [
        koalabear.ExtendedKoalaBear(hash(randomness, koalabear.KoalaBear(31337+i)).value[:4])
        for i in range(log2(count))
    ]

def generate_weights(randomness, count, hash):
    return chi_weights(generate_weights_seed_coords(randomness, count, hash))

def hash16_to_8(inp, permutation):
    return permutation(inp)[...,:8] + inp[...,:8]

def hash(*args, permutation):
    inputs = []
    for arg in args:
        inputs.append(koalabear.KoalaBear(arg.value.reshape((-1,))))
    buffer = koalabear.KoalaBear.append(*inputs)
    from hashlib import sha256
    d = sha256(buffer.tobytes()).digest()
    return koalabear.KoalaBear([int.from_bytes(d[i:i+4], 'little') for i in range(0,32,4)])
    buffer_length = buffer.shape[0]
    # padding: buffer -> [length] + buffer + [pad to nearest 16]
    buffer = koalabear.KoalaBear.append(
        koalabear.KoalaBear([buffer_length]),
        buffer,
        koalabear.KoalaBear.zeros(15 - buffer.shape[0] % 16)
    )
    # Merkle tree style hash
    while buffer.shape[0] > 8:
        if buffer.shape[0] % 16 == 8:
            buffer = koalabear.KoalaBear.append(buffer, koalabear.KoalaBear.zeros(8)) + buffer_length
        buffer = buffer.reshape((-1, 16))
        buffer = hash16_to_8(buffer, permutation)
        buffer = buffer.reshape((-1,))
    return buffer
