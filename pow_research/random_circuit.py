import random

modulus = 97

def mkrandom(width, length):
    o = []
    for i in range(length):
        o.append((random.randrange(width), random.randrange(width),
                 random.randrange(width)))
    return o

def eval(inp, prog):
    o = [x for x in inp]
    for p in prog:
        out, mul1, mul2 = p
        o[out] = (o[out] + mul1 * mul2) % modulus
    return o

def mkinp(width):
    return [random.randrange(modulus) for i in range(width)]
