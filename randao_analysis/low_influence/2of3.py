# Implements the "iterated 2-of-3 majority" low-influence function
# from https://arxiv.org/pdf/1406.5694.pdf and outputs the probability
# that any specific user will be able to influence the result

import random

def mkbits(depth):
    return random.randrange(2**(3**depth))

def winner(val, depth):
    if depth == 0:
        return val & 1
    subwinners = [
        winner(val, depth-1),
        winner(val >> (3**(depth-1)), depth-1),
        winner(val >> (3**(depth-1) * 2), depth-1)
    ]
    return 1 if sum(subwinners) >= 2 else 0

def is_marginal(val, depth):
    if depth == 0:
        return True
    s1, s2, s3 = val, val >> (3**(depth-1)), val >> (3**(depth-1) * 2)
    w1, w2, w3 = (
        winner(s1, depth-1),
        winner(s2, depth-1),
        winner(s3, depth-1)
    )
    if w1 == w2 and w2 == w3:
        return False

    dominants = (s2, s3) if (w1 != w2 and w1 != w3) else \
                (s1, s3) if (w2 != w1 and w2 != w3) else \
                (s1, s2) if (w3 != w1 and w3 != w2) else \
                False

    return is_marginal(dominants[0], depth-1) or \
           is_marginal(dominants[1], depth-1)

def is_marginal2(val, depth):
    w = winner(val, depth)
    for x in range(3**depth):
        val2 = val ^ (1 << x)
        w2 = winner(val2, depth)
        if w2 != w:
            return True
    return False

def influence(val, depth):
    tot = 0
    w = winner(val, depth)
    for i in range(3**depth):
        val2 = val ^ (1 << i)
        w2 = winner(val2, depth)
        if w != w2:
            tot += 1
    return tot / (3**depth)

bitz = [mkbits(3) for i in range(1000)]

print(sum([influence(b, 3) for b in bitz]) / 1000)
