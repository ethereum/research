from ethereum.utils import safe_ord as ord

# 0100000101010111010000110100100101001001 -> ASCII
def decode_bin(x):
    o = bytearray(len(x) // 8)
    for i in range(0, len(x), 8):
        v = 0
        for c in x[i:i+8]:
            v = v * 2 + c
        o[i//8] = v
    return bytes(o)


# ASCII -> 0100000101010111010000110100100101001001
def encode_bin(x):
    o = b''
    for c in x:
        c = ord(c)
        p = bytearray(8)
        for i in range(8):
            p[7-i] = c % 2
            c //= 2
        o += p
    return o

two_bits = [bytes([0,0]), bytes([0,1]),
            bytes([1,0]), bytes([1,1])]
prefix00 = bytes([0,0])
prefix100000 = bytes([1,0,0,0,0,0])


# Encodes a sequence of 0s and 1s into tightly packed bytes
def encode_bin_path(b):
    b2 = bytes((4 - len(b)) % 4) + b
    prefix = two_bits[len(b) % 4]
    if len(b2) % 8 == 4:
        return decode_bin(prefix00 + prefix + b2)
    else:
        return decode_bin(prefix100000 + prefix + b2)


# Decodes bytes into a sequence of 0s and 1s
def decode_bin_path(p):
    p = encode_bin(p)
    if p[0] == 1:
        p = p[4:]
    assert p[0:2] == prefix00
    L = two_bits.index(p[2:4])
    return p[4+((4 - L) % 4):]

def common_prefix_length(a, b):
    o = 0
    while o < len(a) and o < len(b) and a[o] == b[o]:
        o += 1
    return o
        
