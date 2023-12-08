from web3 import Web3
import requests
import sys
if len(sys.argv) >= 2:
    APIKEY = sys.argv[1]
else:
    print("Please get an API key from polygonscan and pass it as an argument")
    sys.exit()

ADDRESS = "0x5FF137D4b0FDCD49DcA30c7CF57E578a026d2789"

PREFIX = bytes.fromhex('1fad948c')

ABI = [
    {'inputs': [{'components': [{'internalType': 'address', 'name': 'sender', 'type': 'address'}, {'internalType': 'uint256', 'name': 'nonce', 'type': 'uint256'}, {'internalType': 'bytes', 'name': 'initCode', 'type': 'bytes'}, {'internalType': 'bytes', 'name': 'callData', 'type': 'bytes'}, {'internalType': 'uint256', 'name': 'callGasLimit', 'type': 'uint256'}, {'internalType': 'uint256', 'name': 'verificationGasLimit', 'type': 'uint256'}, {'internalType': 'uint256', 'name': 'preVerificationGas', 'type': 'uint256'}, {'internalType': 'uint256', 'name': 'maxFeePerGas', 'type': 'uint256'}, {'internalType': 'uint256', 'name': 'maxPriorityFeePerGas', 'type': 'uint256'}, {'internalType': 'bytes', 'name': 'paymasterAndData', 'type': 'bytes'}, {'internalType': 'bytes', 'name': 'signature', 'type': 'bytes'}], 'internalType': 'struct UserOperation[]', 'name': 'ops', 'type': 'tuple[]'}, {'internalType': 'address payable', 'name': 'beneficiary', 'type': 'address'}], 'name': 'handleOps', 'outputs': [], 'stateMutability': 'nonpayable', 'type': 'function'}
]
FIELDS = ['sender', 'nonce', 'callGasLimit', 'verificationGasLimit', 'preVerificationGas', 'maxFeePerGas', 'maxPriorityFeePerGas', 'paymasterAndData', 'initCode', 'callData', 'signature']
NUMBER_FIELDS = ['nonce', 'callGasLimit', 'verificationGasLimit', 'preVerificationGas', 'maxFeePerGas', 'maxPriorityFeePerGas']
ADDRESS_FIELDS = ['sender']
BYTES_FIELDS = {'initCode': 32, 'callData': 4, 'paymasterAndData': 32, 'signature': 32}

w3 = Web3()
contract = w3.eth.contract(abi=ABI)

def fetch_transactions(address):
    url = f"https://api.polygonscan.com/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={APIKEY}"

    response = requests.get(url)

    if response.status_code == 200:
        transactions = response.json().get("result", [])
        calldata = [bytes.fromhex(tx["input"][2:]) for tx in transactions]
        return calldata
    else:
        return f"Error: {response.status_code}"

def split32(data, first_chunk_length):
    o = []
    for i in range(first_chunk_length, len(data) + 32, 32):
        o.append(data[max(0, i-32): min(i, len(data))])
    return o

FIELDS_EXT = FIELDS + ["beneficiary"]

calldatas = fetch_transactions(ADDRESS)
freqs = {field: {} for field in FIELDS_EXT}
for calldata in calldatas:
    if calldata[:4] == PREFIX:
        decoded = contract.decode_function_input(calldata)
        for op in decoded[1]['ops']:
            for key, value in op.items():
                if key in BYTES_FIELDS:
                    offset = BYTES_FIELDS[key]
                    for chunk in split32(value, offset):
                        freqs[key][chunk] = freqs[key].get(chunk, 0) + 1
                else:
                    freqs[key][value] = freqs[key].get(value, 0) + 1
        b = decoded[1]["beneficiary"]
        freqs["beneficiary"][b] = freqs["beneficiary"].get(b, 0) + 1

dictionaries = {field: [] for field in FIELDS_EXT}
topfreqs = {field: [] for field in FIELDS_EXT}
for field in FIELDS_EXT:
    for k in sorted(list(freqs[field].keys()), key=lambda k: -freqs[field][k])[:192]:
        dictionaries[field].append(k)
        topfreqs[field].append(freqs[field][k])

def compress_bytes(data, offset, dictionary):
    bits = 0
    chunks = split32(data, offset)
    o = []
    #print("Data is {} bytes ({} chunks)".format(len(data), len(chunks)))
    for i, chunk in enumerate(chunks):
        if chunk in dictionary:
            o.append(bytes([dictionary.index(chunk)]))
            bits += 2**i
        elif len(chunk) == 32 and chunk[0] == 0:
            stripped_chunk = chunk.lstrip(b'\x00')
            assert stripped_chunk != b''
            o.append(bytes([224 - len(stripped_chunk)]) + stripped_chunk)
            bits += 2**i
        elif len(chunk) == 32 and chunk[-1] == 0:
            stripped_chunk = chunk.rstrip(b'\x00')
            assert stripped_chunk != b''
            o.append(bytes([256 - len(stripped_chunk)]) + stripped_chunk)
            bits += 2**i
        else:
            o.append(chunk)
        #print("Chunk {}: {}".format(i, chunk.hex()))
    bitfield_bytes = (len(chunks) + 7) // 8
    prefix = (
        bytes([bitfield_bytes]) +
        bits.to_bytes(bitfield_bytes, 'little')
    )
    output = prefix + b''.join(o)
    #print("Pre_output length: {}".format(2 + len(prefix)))
    return len(output).to_bytes(2, 'little') + output

def decompress_bytes(data, offset, dictionary, pos=0):
    length = int.from_bytes(data[pos: pos+2], 'little')
    pos += 2
    bitfield_length = data[pos]
    bitfield = int.from_bytes(data[pos+1: pos+1+bitfield_length], 'little')
    o = []
    subdata_end = pos + length
    pos += 1 + bitfield_length
    #print("Pre_output length: {}".format(2 + 1 + bitfield_length))
    while pos < subdata_end:
        if bitfield % 2:
            if data[pos] < 192:
                o.append(dictionary[data[pos]])
                pos += 1
            elif data[pos] < 224:
                zeros = data[pos] - 192
                o.append(b'\x00' * zeros + data[pos+1: pos+33-zeros])
                pos += 33 - zeros
            else:
                zeros = data[pos] - 224
                o.append(data[pos+1: pos+33-zeros] + b'\x00' * zeros)
                pos += 33 - zeros
        else:
            o.append(data[pos: min(pos+offset, subdata_end)])
            pos += offset
        offset = 32
        #print("Chunk: {}".format(o[-1].hex()))
        bitfield //= 2
        front_crop = False
    return b''.join(o), subdata_end

def compress(data, dictionaries):
    o = []
    try:
        decoded = contract.decode_function_input(calldata)
    except:
        return b'\xff' + data
    b = decoded[1]["beneficiary"]
    if b in dictionaries["beneficiary"]:
        o.append(bytes([dictionaries["beneficiary"].index(b)]))
    else:
        o.append(b'\xfe' + bytes.fromhex(b[2:]))
    for op in decoded[1]["ops"]:
        #print("Bitfield: [2 bytes]")
        bits = 0
        op_output = []
        for i, field in enumerate(FIELDS[:-4]): # skip callData and signature for now
            val = op[field]
            if val in dictionaries[field]:
                op_output.append(bytes([dictionaries[field].index(val)]))
                bits += 2**i
            elif field in NUMBER_FIELDS:
                length = (val.bit_length() + 7) // 8
                op_output.append(bytes([length]) + val.to_bytes(length, 'little'))
            elif field in ADDRESS_FIELDS:
                assert val[:2] == '0x'
                op_output.append(bytes.fromhex(val[2:]))
            else:
                raise Exception("wat")
            #print("{}: {} ({} bytes)".format(field, op_output[-1].hex(), len(op_output[-1])))
        o.append(bits.to_bytes(2, 'little'))
        #print("Bitfield is: {}".format(bin(bits)[2:]))
        o.extend(op_output)
        for field, offset in BYTES_FIELDS.items():
            #print("Compressing {}".format(field))
            o.append(compress_bytes(op[field], offset, dictionaries[field]))
            assert decompress_bytes(o[-1], offset, dictionaries[field], 0)[0] == op[field]
            #print("{}: {} ({} bytes)".format(field, o[-1].hex(), len(o[-1])))
    return b''.join(o)

def decompress(data, dictionaries):
    if data[0] == 255:
        return data[1:]
    ops = []
    pos = 0
    if data[0] == 254:
        beneficiary = '0x'+bytes.fromhex(data[1:21])
        pos = 21
    else:
        beneficiary = dictionaries["beneficiary"][data[0]]
        pos = 1
    while pos < len(data):
        #print('Beginning to decompress an op', pos)
        bits = int.from_bytes(data[pos: pos+2], 'little')
        #print("Bitfield is: {}".format(bin(bits)[2:]))
        pos += 2
        args = {}
        for i, field in enumerate(FIELDS[:-4]):
            #print("Decompressing {}, pos {}".format(field, pos))
            if bits & (2**i):
                args[field] = dictionaries[field][data[pos]]
                pos += 1
            else:
                if field in NUMBER_FIELDS:
                    length = data[pos]
                    databytes = data[pos+1: pos+1+length]
                    args[field] = int.from_bytes(databytes, 'little')
                    pos += 1 + length
                elif field in ADDRESS_FIELDS:
                    args[field] = w3.to_checksum_address(
                        "0x" + data[pos: pos+20].hex()
                    )
                    pos += 20
                else:
                    raise Exception("wat")
        for field, offset in BYTES_FIELDS.items():
            #print("Decompressing {}, pos {}".format(field, pos))
            val, pos = decompress_bytes(
                data, offset, dictionaries[field], pos
            )
            args[field] = val
        ops.append(args)
    
    return bytes.fromhex(contract.encodeABI(fn_name='handleOps', args={
        "ops": ops, "beneficiary": w3.to_checksum_address(beneficiary)
    })[2:])

total_original_length, total_compressed_length = 0, 0
for calldata in calldatas:
    total_original_length += len(calldata)
    c = compress(calldata, dictionaries)
    print('Compressed: {}'.format(c.hex()))
    total_compressed_length += len(c)
    d = decompress(c, dictionaries)
    assert d == calldata, (calldata, c, d)
    print("Original: {}, compressed: {}".format(len(d), len(c)))
    #for i in range(4, len(calldata), 32):
    #    print('    '+calldata[i:i+32].hex())

print("Total original: {}, total compressed: {} ({:.2f}x)".format(
    total_original_length,
    total_compressed_length,
    total_original_length / total_compressed_length
))
