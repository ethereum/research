import random

try: #
    shathree = __import__('sha3').sha3_256
except:
    shathree = __import__('python_sha3').sha3_256


params = {
    "size": 256,
    "pecks": 32
}


def decode_int(x):
    o = 0
    for a in x:
        o = o * 256 + ord(a)
    return o


def sha3(x):
    return shathree(x).digest()


def bloom_insert(params, bloom, val):
    k = decode_int(sha3(val)) * (3**160 + 112)
    for i in range(params["pecks"]):
        bloom |= 1 << (k % params["size"])
        k //= params["size"]
    return bloom


def bloom_query(params, bloom, val):
    o = bloom_insert(params, 0, val)
    return (bloom & o) == o


def test_params(size, pecks, objcount):
    params = {"size": size, "pecks": pecks}
    count = 0
    for i in range(100):
        objs = [str(random.randrange(2**40)) for i in range(objcount)]
        bloom = 0
        for o in objs:
            bloom = bloom_insert(params, bloom, o)
    
        for o in objs:
            assert bloom_query(params, bloom, o)
    
        for i in range(100):
            if bloom_query(params, bloom, str(random.randrange(2**40))):
                count += 1
    
    print 'False positive rate: %f' % (count / 10000.)
