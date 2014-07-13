import pyethereum
u = pyethereum.utils
import time

ops = [
    lambda x, y: x+y % 2**256,
    lambda x, y: x*y % 2**256,
    lambda x, y: x % y if y > 0 else x+y,
    lambda x, y: x & y,
    lambda x, y: x | y,
    lambda x, y: x ^ y
]


def gen(seed, w, d):
    tape = []
    h = 0
    for i in range(d):
        if h < 2**32:
            h = u.big_endian_to_int(u.sha3(seed+str(i)))
        v1 = h % w
        h /= w
        v2 = h % w
        h /= w
        op = ops[h % len(ops)]
        h /= len(ops)
        tape.append([v1, v2, op])
    return tape


def lshift(n):
    return 2**255 * (n % 2) + (n / 2)


def run(seed, w, tape):
    v = []
    h = 0
    for i in range(w):
        if i % 1 == 0:
            h = u.big_endian_to_int(u.sha3(seed+str(i)))
        else:
            h = lshift(h)
        v.append(h)
    for t in tape:
        v[t[0]] = t[2](v[t[0]], v[t[1]])
    return u.sha3(str(v))


def test():
    t1 = time.time()
    for i in range(10):
        tape = gen(str(i), 1000, 1000)
    print time.time() - t1

    t2 = time.time()
    for i in range(10):
        p = run(str(i), 1000, tape)
    print time.time() - t2
    p = p
