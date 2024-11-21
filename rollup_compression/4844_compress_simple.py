import requests
import sys

PREFIX = bytes.fromhex('1fad948c')
if len(sys.argv) >= 2:
    APIKEY = sys.argv[1]
else:
    print("Please get an API key from polygonscan and pass it as an argument")
    sys.exit()

def fetch_transactions(address):
    url = f"https://api.polygonscan.com/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={APIKEY}"

    response = requests.get(url)

    if response.status_code == 200:
        transactions = response.json().get("result", [])
        calldata = [bytes.fromhex(tx["input"][2:]) for tx in transactions]
        return calldata
    else:
        return f"Error: {response.status_code}"

# Replace with the actual address you're interested in
address = "0x5FF137D4b0FDCD49DcA30c7CF57E578a026d2789"
calldatas = fetch_transactions(address)
freqs = {}
for calldata in calldatas:
    for i in range(4, len(calldata), 32):
        chunk = calldata[i:i+32]
        freqs[chunk] = freqs.get(chunk, 0) + 1

dictionary = []
for k in sorted(list(freqs.keys()), key=lambda k: -freqs[k])[:192]:
    print(k.hex(), freqs[k])
    dictionary.append(k)

def compress(data, dictionary):
    if data[:4] != PREFIX or len(data) % 32 != 4:
        return b'\xff' + data
    bitfield = 0
    output = []
    chunk_count = len(data) // 32
    for i in range(chunk_count):
        chunk = data[4 + i*32: 36 + i*32]
        if chunk in dictionary:
            output.append(bytes([dictionary.index(chunk)]))
            bitfield += 2**i
        elif chunk[0] == 0:
            stripped_chunk = chunk.lstrip(b'\x00')
            output.append(bytes([224 - len(stripped_chunk)]) + stripped_chunk)
            bitfield += 2**i
        elif chunk[-1] == 0:
            stripped_chunk = chunk.rstrip(b'\x00')
            output.append(bytes([256 - len(stripped_chunk)]) + stripped_chunk)
            bitfield += 2**i
        else:
            output.append(chunk)
    assert (chunk_count + 7) // 8 < 255
    return (
        bytes([(chunk_count + 7) // 8]) +
        bitfield.to_bytes((chunk_count + 7) // 8, 'little') +
        b''.join(output)
    )

def decompress(data, dictionary):
    if data[0] == 255:
        return data[1:]
    chunk_offset = 1 + data[0]
    bitfield = int.from_bytes(data[1: chunk_offset], 'little')
    chunks = [data[i: i+32] for i in range(chunk_offset, len(data), 32)]
    output = []
    pos = chunk_offset
    for i in range(data[0] * 8):
        if pos >= len(data):
            break
        if bitfield & (2**i):
            if data[pos] < 192:
                output.append(dictionary[data[pos]])
                pos += 1
            elif data[pos] < 224:
                zeros = data[pos] - 192
                output.append(b'\x00' * zeros + data[pos+1: pos+33-zeros])
                pos += 33 - zeros
            else:
                zeros = data[pos] - 224
                output.append(data[pos+1: pos+33-zeros] + b'\x00' * zeros)
                pos += 33 - zeros
        else:
            output.append(data[pos: pos+32])
            pos += 32
    return PREFIX + b''.join(output)

total_original_length, total_compressed_length = 0, 0
for calldata in calldatas:
    total_original_length += len(calldata)
    c = compress(calldata, dictionary)
    total_compressed_length += len(c)
    d = decompress(c, dictionary)
    assert d == calldata, (calldata, c, d)
    print("Original: {}, compressed: {}".format(len(d), len(c)))
    #for i in range(4, len(calldata), 32):
    #    print('    '+calldata[i:i+32].hex())

print("Total original: {}, total compressed: {} ({:.2f}x)".format(
    total_original_length,
    total_compressed_length,
    total_original_length / total_compressed_length
))
