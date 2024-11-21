import snappy

data = bytes.fromhex('0x000000000000000000000000a5548cb22dadac786972a2a91e55af6b4209563a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000001a0000000000000000000000000000000000000000000000000000000000000f12800000000000000000000000000000000000000000000000000000000000186a0000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000002540be40000000000000000000000000000000000000000000000000000000002540be4000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000026000000000000000000000000000000000000000000000000000000000000002800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008480c5c7d0000000000000000000000000febebb892587ecf190f3b948dd1dcb60c9679b3400000000000000000000000000000000000000000000000000038d7ea4c6800000000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004172b8b9720baf0341264ec195e2550393772c222b10cff626c0863aafd98921516af2320acce7e49b6075bbb5a3ac0ae8dc9555f7e70924a53aedd57f050abe331b00000000000000000000000000000000000000000000000000000000000000'[2:])

def zrle_compress(inp):
    o = []
    pos = 0
    while pos < len(inp):
        zcount = 0
        while pos + zcount < len(inp) and inp[pos + zcount] == 0 and zcount < 255:
            zcount += 1
        if zcount > 0:
            pos += zcount
            o.extend([0, zcount])
        else:
            o.append(inp[pos])
            pos += 1
    return bytes(o)

def zrle_decompress(inp):
    o = []
    pos = 0
    while pos < len(inp):
        if inp[pos] != 0:
            o.append(inp[pos])
            pos += 1
        else:
            o.extend([0] * inp[pos + 1])
            pos += 2
    return bytes(o)

def ctl_compress(inp):
    assert len(inp) % 32 == 0
    o = []
    for pos in range(0, len(inp), 32):
        chunk = inp[pos: pos+32]
        l_stripped_chunk = chunk.lstrip(b'\x00')
        r_stripped_chunk = chunk.rstrip(b'\x00')
        if len(l_stripped_chunk) < len(r_stripped_chunk):
            o.append(len(l_stripped_chunk))
            o.extend(l_stripped_chunk)
        else:
            o.append(128 + len(r_stripped_chunk))
            o.extend(r_stripped_chunk)
    return bytes(o)

def ctl_decompress(inp):
    o = []
    pos = 0
    while pos < len(inp):
        chunk_length = inp[pos]
        stripped_chunk = inp[pos + 1: pos + 1 + (chunk_length % 128)]
        if chunk_length < 128:
            o.extend([0] * (32 - chunk_length))
            o.extend(stripped_chunk)
        else:
            o.extend(stripped_chunk)
            o.extend([0] * (160 - chunk_length))
        pos += 1 + (chunk_length % 128)
    return bytes(o)

def gascost(data):
    return len(data) * 16 - 12 * data.count(b'\x00')

print("Raw data: length {} gas cost {}".format(len(data), gascost(data)))

#for i in range(0, len(data), 32):
#    print(data[i:i+32].hex())

zdata = zrle_compress(data)
assert data == zrle_decompress(zdata)
print("ZRLE: length {} gas cost {}".format(len(zdata), gascost(zdata)))

cdata = ctl_compress(data)
assert data == ctl_decompress(cdata)
print("CTL: length {} gas cost {}".format(len(cdata), gascost(cdata)))

sdata = snappy.compress(data)
print("Snappy: length {} gas cost {}".format(len(sdata), gascost(sdata)))
