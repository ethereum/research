# The work here is outdated, please ignore

import urllib.request, json

url = 'https://api.etherscan.io/api?module=account&action=txlist&address=0x5e4e65926ba27467555eb562121fac00d24e9dd2&startblock=0&endblock=99999999&page=1&offset=100&sort=desc&apikey=YourApiKeyToken'
j = json.loads(urllib.request.urlopen(url).read())['result']
data = b''.join([bytes.fromhex(x['input'][2:]) for x in j])
print("Raw data length: {}".format(len(data)))

import zlib

print("Zlib compress length: {}".format(len(zlib.compress(data))))

def zbyte_compress(data):
    o = []
    i = 0
    while i < len(data):
        if data[i] == 254:
            o.extend([254, 0])
        elif list(data[i:i+2]) == [0, 0]:
            p = 2
            while p < 255 and i + p < len(data) and data[i + p] == 0:
                p += 1
            o.extend([254, p])
            i += p - 1
        else:
            o.append(data[i])
        i += 1
    return bytes(o)

def zbyte_decompress(data):
    o = []
    i = 0
    while i < len(data):
        if data[i] == 254:
            if i == len(data) - 1:
                raise Exception("Invalid encoding, \\xfe at end")
            elif data[i+1] == 0:
                o.append(254)
            else:
                o.extend([0] * data[i+1])
            i += 1
        else:
            o.append(data[i])
        i += 1
    return bytes(o)

c = zbyte_compress(data)

assert zbyte_decompress(c) == data

print("Zero-byte compress length: {}".format(len(c)))

def simple_compress(data, chunk_size=28):
    q = data.replace(b'\xfe', b'\xfe\xff')
    winning_keys = []
    for i in range(200):
        occurrences = {}
        for pos in range(0, len(q)-chunk_size+1):
            chunk = q[pos:pos+chunk_size]
            occurrences[chunk] = occurrences.get(chunk, 0) + 1
        winning_key = max(occurrences.keys(), key=lambda x: occurrences[x])
        winning_keys.append(winning_key)
        q = q.replace(winning_key, bytes([254, i]))
        print("Key: {}, occurrences: {}, new length: {}"
              .format(winning_key, occurrences[winning_key], len(q) + chunk_size * (i+1)))
    return(b''.join(winning_keys) + q)

print("Simple compress length: {}".format(len(simple_compress(zbyte_compress(data)))))
