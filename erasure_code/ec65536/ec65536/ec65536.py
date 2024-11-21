import copy
import poly_utils
import rlp

try:
    from Crypto.Hash import keccak
    sha3 = lambda x: keccak.new(digest_bits=256, data=x).digest()
except ImportError:
    import sha3 as _sha3
    sha3 = lambda x: _sha3.sha3_256(x).digest()

# Every point is an element of GF(2**16), so represents two bytes
POINT_SIZE = 2
# Every chunk contains 128 points
POINTS_IN_CHUNK = 128
# A chunk is 256 bytes
CHUNK_SIZE = POINT_SIZE * POINTS_IN_CHUNK

def bytes_to_num(bytez):
    o = 0
    for b in bytez:
        o = (o * 256) + b
    return o

def num_to_bytes(inp, n):
    o = b''
    for i in range(n):
        o = bytes([inp % 256]) + o
        inp //= 256
    return o

assert bytes_to_num(num_to_bytes(31337, 2)) == 31337

# Returns the smallest power of 2 equal to or greater than a number
def higher_power_of_2(x):
    higher_power_of_2 = 1
    while higher_power_of_2 < x:
        higher_power_of_2 *= 2
    return higher_power_of_2

# Unfortunately, most padding schemes standardized in cryptography seem to only work for
# block sizes strictly less than 256 bytes. So we'll use RLP plus zero byte padding
# instead (pre-RLP-encode because the RLP encoding adds length data, so the padding
# becomes reversible even in cases where the original data ends in zero bytes)
def pad(data):
    med = rlp.encode(data)
    return med + b'\x00' * (higher_power_of_2(len(med)) - len(med))

def unpad(data):
    c, l1, l2 = rlp.codec.consume_length_prefix(data)
    assert c == str
    return data[:l1 + l2]

# Deserialize a chunk into a list of points in GF2**16
def chunk_to_points(chunk):
    return [bytes_to_num(chunk[i: i + POINT_SIZE]) for i in range(0, CHUNK_SIZE, POINT_SIZE)]

# Serialize a list of points into a chunk
def points_to_chunk(points):
    return b''.join([num_to_bytes(p, POINT_SIZE) for p in points])

testdata = sha3(b'cow') * (CHUNK_SIZE // 32)
assert points_to_chunk(chunk_to_points(testdata)) == testdata

# Make a Merkle tree out of a set of chunks
def merklize(chunks):
    # Only accept a list of size which is exactly a power of two
    assert higher_power_of_2(len(chunks)) == len(chunks)
    merkle_nodes = [sha3(x) for x in chunks]
    lower_tier = merkle_nodes[::]
    higher_tier = []
    while len(higher_tier) != 1:
        higher_tier = [sha3(lower_tier[i] + lower_tier[i + 1]) for i in range(0, len(lower_tier), 2)]
        merkle_nodes = higher_tier + merkle_nodes
        lower_tier = higher_tier
    merkle_nodes.insert(0, b'\x00' * 32)
    return merkle_nodes


class Prover():
    def __init__(self, data):
        # Pad data
        pdata = pad(data)
        byte_chunks = [pdata[i: i + CHUNK_SIZE] for i in range(0, len(pdata), CHUNK_SIZE)]
        # Decompose it into chunks, where each chunk is a collection of numbers
        chunks = []
        for byte_chunk in byte_chunks:
            chunks.append(chunk_to_points(byte_chunk))
        # Compute the polynomials representing the ith number in each chunk
        polys = [poly_utils.lagrange_interp([chunk[i] for chunk in chunks], list(range(len(chunks)))) for i in range(POINTS_IN_CHUNK)]
        # Use the polynomials to extend the chunks
        new_chunks = []
        for x in range(len(chunks), len(chunks) * 2):
            new_chunks.append(points_to_chunk([poly_utils.eval_poly_at(poly, x) for poly in polys]))
        # Total length of data including new points
        self.length = len(byte_chunks + new_chunks)
        self.extended_data = byte_chunks + new_chunks
        # Build up the Merkle tree
        self.merkle_nodes = merklize(self.extended_data)
        assert len(self.merkle_nodes) == 2 * self.length
        self.merkle_root = self.merkle_nodes[1]

    # Make a Merkle proof for some index
    def prove(self, index):
        assert 0 <= index < self.length
        adjusted_index = self.length + index
        o = [self.extended_data[index]]
        while adjusted_index > 1:
            o.append(self.merkle_nodes[adjusted_index ^ 1])
            adjusted_index >>= 1
        return o

# Verify a merkle proof of some index (light client friendly)
def verify_proof(merkle_root, proof, index):
    h = sha3(proof[0])
    for p in proof[1:]:
        if index % 2:
            h = sha3(p + h)
        else:
            h = sha3(h + p)
        index //= 2
    return h == merkle_root

# Fill data from partially available proofs
# This method returning False can also be used as a verifier for fraud proofs
def fill(merkle_root, orig_data_length, proofs, indices):
    if len(proofs) < orig_data_length:
        raise Exception("Not enough proofs")
    if len(proofs) > orig_data_length:
        raise Exception("Too many proofs; if original data has n chunks, n chunks suffice")
    for proof, index in zip(proofs, indices):
        if not verify_proof(merkle_root, proof, index):
            raise Exception("Merkle proof for index %d invalid" % index)
    # Convert to points
    coords = [chunk_to_points(p[0]) for p in proofs]
    # Extract polynomials
    polys = [poly_utils.lagrange_interp([c[i] for c in coords], indices) for i in range(POINTS_IN_CHUNK)]
    # Fill in the remaining values
    full_coords = [None] * orig_data_length * 2
    for points, index in zip(coords, indices):
        full_coords[index] = points
    for i in range(len(full_coords)):
        if full_coords[i] is None:
            full_coords[i] = [poly_utils.eval_poly_at(poly, i) for poly in polys]
    # Serialize
    full_chunks = [points_to_chunk(points) for points in full_coords]
    # Merklize
    merkle_nodes = merklize(full_chunks)
    # Check equality of the Merkle root
    if merkle_root != merkle_nodes[1]:
        return False
    return full_chunks
