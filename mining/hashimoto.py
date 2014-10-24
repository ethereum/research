try:
    shathree = __import__('sha3')
except:
    shathree = __import__('python_sha3')
import random
import time


def sha3(x):
    return decode_int(shathree.sha3_256(x).digest()) #


def decode_int(s):
    o = 0
    for i in range(len(s)):
        o = o * 256 + ord(s[i])
    return o


def encode_int(x):
    o = ''
    for _ in range(32):
        o = chr(x % 256) + o
        x //= 256
    return o

P = 2**256 - 4294968273


def produce_dag(params, seed):
    o = [sha3(seed)]
    init = o[0]
    picker = 1
    for i in range(1, params["n"]):
        x = 0
        picker = (picker * init) % P
        curpicker = picker
        for j in range(params["k"]):
            x |= o[curpicker % i]
            curpicker >>= 10
        o.append((x * x) % P)  # use any "hash function" here
    return o


def quick_calc(params, seed, pos):
    init = sha3(seed)
    known = {0: init}

    def calc(p):
        if p not in known:
            picker = pow(init, p, P)
            x = 0
            for j in range(params["k"]):
                x |= calc(picker % p)
                picker >>= 10
            known[p] = (x * x) % P
        return known[p]

    o = calc(pos)
    print 'Calculated pos %d with %d accesses' % (pos, len(known))
    return o


def hashimoto(daggerset, params, header, nonce):
    rand = sha3(header+encode_int(nonce))
    mix = 0
    for i in range(40):
        shifted_A = rand >> i
        dag = daggerset[shifted_A % params["numdags"]]
        mix ^= dag[(shifted_A // params["numdags"]) % params["n"]]
    return mix ^ rand


def get_daggerset(params, block):
    if block.number == 0:
        return [produce_dag(params, i) for i in range(params["numdags"])]
    elif block.number % params["epochtime"]:
        return get_daggerset(block.parent)
    else:
        o = get_daggerset(block.parent)
        o[sha3(block.parent.nonce) % params["numdags"]] = \
            produce_dag(params, sha3(block.parent.nonce))
        return o


def mine(daggerset, params, header):
    nonce = random.randrange(2**50)
    orignonce = nonce
    origtime = time.time()
    while 1:
        h = hashimoto(daggerset, params, header, nonce)
        if h <= 2**256 / params["diff"]:
            noncediff = nonce - orignonce
            timediff = time.time() - origtime
            print 'Found nonce: %d, tested %d nonces in %f seconds (%f per sec)' % \
                (nonce, noncediff, timediff, noncediff / timediff)
            return nonce
        nonce += 1


def verify(daggerset, params, header, nonce):
    return hashimoto(daggerset, params, header, nonce) \
        <= 2**256 / params["diff"]


def light_hashimoto(seedset, params, header, nonce):
    rand = sha3(header+encode_int(nonce))
    mix = 0
    for i in range(40):
        shifted_A = rand >> i
        seed = seedset[shifted_A % params["numdags"]]
        # can further optimize with cross-round memoization
        mix ^= quick_calc(params, seed,
                          (shifted_A // params["numdags"]) % params["n"])
    return mix ^ rand


def light_verify(seedset, params, header, nonce):
    return light_hashimoto(seedset, params, header, nonce) \
        <= 2**256 / params["diff"]

params = {
    "numdags": 40,
    "n": 250000,
    "diff": 2**14,
    "epochtime": 100,
    "k": 3
}
