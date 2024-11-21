# Implements a modified version of the TRIBES low-influence function
# mentioned in https://arxiv.org/pdf/1406.5694.pdf and outputs the
# probability that any specific user will be able to influence the result

import random, math

def mkbits(n):
    return random.randrange(2**n)

def tribes_log(n):
    w = 1
    while w * 2**w * 693 < n * 1000:
        w += 1
    return w

def tribes(val, n):
    split = tribes_log(n)
    o = []
    full_subset = (1 << split) - 1
    for i in range(n):
        vall = val ^ ((2*i+3)**n % 2**n)
        t = 0
        for _ in range(n // split):
            if vall & full_subset == full_subset:
                t = 1
                break
            vall >>= split
        o.append(t)
        if len(o) % 2 == 0 and o[-2] == 0 and o[-1] == 1:
            return 0
        if len(o) % 2 == 0 and o[-2] == 1 and o[-1] == 0:
            return 1
    return o[-1]

def influence(val, n):
    tot = 0
    w = tribes(val, n)
    for i in range(n):
        val2 = val ^ (1 << i)
        w2 = tribes(val2, n)
        if w != w2:
            tot += 1
    return tot / n

print(sum([influence(mkbits(50), 50) for i in range(1000)]) / 1000)
# print(sum([tribes(mkbits(50), 50) for i in range(1000)]) / 1000)
