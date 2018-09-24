import random

def mk_shuffle(n):
    L = list(range(n))
    random.shuffle(L)
    return L

def mk_fragmented_shuffle(n, shuffs):
    L = list(range(n))
    for i in range(shuffs):
        x1 = random.randrange(n)
        x2 = random.randrange(n)
        value = int(min(n - x1, n - x2, abs(x2 - x1)) ** random.random())
        L[x1:x1+value], L[x2:x2+value] = L[x2:x2+value], L[x1:x1+value]
    return L

def fragments(vals):
    tot = 1
    for i in range(1, len(vals)):
        if vals[i] != vals[i-1] + 1:
            tot += 1
    return tot

def apply_perm(vals, perm):
    o = [0 for x in vals]
    for i in range(len(perm)):
        o[i] = vals[perm[i]]
    return o

def attempt_fix(vals):
    perm = list(range(len(vals)))
    indices = {}
    for i, x in enumerate(vals):
        indices[x] = i
    for i in range(len(vals)):
        if perm[i] == i and vals[i] != i:
            poz = indices[i]
            if perm[poz] == poz:
                perm[i], perm[poz] = perm[poz], perm[i]
    assert apply_perm(perm, perm) == list(range(len(vals)))
    return perm

def fix(vals):
    goal = list(range(len(vals)))
    path = []
    while vals != goal:
        vals = apply_perm(vals, attempt_fix(vals))
        path.append(vals)
    return path
