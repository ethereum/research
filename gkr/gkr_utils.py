from zorch import koalabear
from utils import hash

#M = koalabear.KoalaBear([[3,1,4,1],[5,9,2,6],[3,5,8,9],[7,9,3,2]])
M = 1 / koalabear.KoalaBear([[1+i+j for i in range(16)] for j in range(16)])

def matmul_layer(values):
    size = values.shape[0] // 16
    values = values.reshape((size, 16))
    values = koalabear.matmul(values, M)
    values = values.reshape((size * 16,))
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

def generate_weights_seed_coords(randomness, count):
    return [
        koalabear.ExtendedKoalaBear(hash(randomness, koalabear.KoalaBear(31337+i)).value[:4])
        for i in range(log2(count))
    ]

def generate_weights(randomness, count):
    return chi_weights(generate_weights_seed_coords(randomness, count))
