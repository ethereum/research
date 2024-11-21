from pyethereum import utils
import random


def sha3(x):
    return utils.decode_int(utils.zunpad(utils.sha3(x)))


class SeedObj():
    def __init__(self, seed):
        self.seed = seed
        self.a = 3**160
        self.c = 7**80
        self.n = 2**256 - 4294968273 # secp256k1n, why not

    def rand(self, r):
        self.seed = (self.seed * self.a + self.c) % self.n
        return self.seed % r


def encode_int(x):
    o = ''
    for _ in range(8):
        o = chr(x % 256) + o
        x //= 256
    return o


ops = {
    "plus": lambda x, y: (x + y) % 2**64,
    "times": lambda x, y: (x * y) % 2**64,
    "xor": lambda x, y: x ^ y,
    "and": lambda x, y: x & y,
    "or": lambda x, y: x | y,
    "not": lambda x, y: 2**64-1-x,
    "nxor": lambda x, y: (2**64-1-x) ^ y,
    "rshift": lambda x, y: x >> (y % 64)
}


def gentape(W, H, SEED):
    s = SeedObj(SEED)
    tape = []
    for i in range(H):
        op = ops.keys()[s.rand(len(ops))]
        r = s.rand(100)
        if r < 20 and i > 20:
            x1 = tape[-r]["x1"]
        else:
            x1 = s.rand(W)
        x2 = s.rand(W)
        tape.append({"op": op, "x1": x1, "x2": x2})
    return tape


def runtape(TAPE, SEED, params):
    s = SeedObj(SEED)
    # Fill up tape inputs
    mem = [0] * params["w"]
    for i in range(params["w"]):
        mem[i] = s.rand(2**64)
    # Direction variable (run forwards or backwards)
    dir = 1
    # Run the tape on the inputs
    for i in range(params["h"] // 100):
        for j in (range(100) if dir == 1 else range(99, -1, -1)):
            t = TAPE[i * 100 + j]
            mem[t["x1"]] = ops[t["op"]](mem[t["x1"]], mem[t["x2"]])
        # 16% of the time, we flip the order of the next 100 ops;
        # this adds in conditional branching
        if 2 < mem[t["x1"]] % 37 < 9:
            dir *= -1
    return sha3(''.join(encode_int(x) for x in mem))


def PoWVerify(header, nonce, params):
    tape = gentape(params["w"], params["h"],
                   sha3(header + encode_int(nonce // params["n"])))
    h = runtape(tape, sha3(header + encode_int(nonce)), params)
    print h
    return h < 2**256 / params["diff"]


def PoWRun(header, params):
    # actual randomness, so that miners don't overlap
    nonce = random.randrange(2**50) * params["n"]
    tape = None
    while 1:
        print nonce
        if nonce % params["n"] == 0:
            tape = gentape(params["w"], params["h"],
                           sha3(header + encode_int(nonce // params["n"])))
        h = runtape(tape, sha3(header + encode_int(nonce)), params)
        if h < 2**256 / params["diff"]:
            return nonce
        else:
            nonce += 1

params = {
    "w": 100,
    "h": 15000,  # generally, w*log(w) at least
    "diff": 2**24,  # initial
    "n": 1000
}
