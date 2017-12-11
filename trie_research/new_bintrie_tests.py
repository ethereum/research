from new_bintrie import Trie, EphemDB, encode_bin, encode_bin_path, decode_bin_path
from ethereum.utils import sha3, encode_hex
import random
import rlp

def shuffle_in_place(x):
    y = x[::]
    random.shuffle(y)
    return y

kvpairs = [(sha3(str(i))[12:], str(i).encode('utf-8') * 5) for i in range(2000)]


for path in ([], [1,0,1], [0,0,1,0], [1,0,0,1,0], [1,0,0,1,0,0,1,0], [1,0] * 8):
    assert decode_bin_path(encode_bin_path(bytes(path))) == bytes(path)

r1 = None

for _ in range(3):
    t = Trie(EphemDB(), b'')
    for i, (k, v) in enumerate(shuffle_in_place(kvpairs)):
        #print(t.to_dict())
        t.update(k, v)
        assert t.get(k) == v
        if not i % 50:
            if not i % 250:
                t.to_dict()
            print("Length of long-format branch at %d nodes: %d" % (i, len(rlp.encode(t.get_long_format_branch(k)))))
    print('Added 1000 values, doing checks')
    assert r1 is None or t.root == r1
    r1 = t.root
    t.update(kvpairs[0][0], kvpairs[0][1])
    assert t.root == r1
    print(encode_hex(t.root))
    print('Checking that single-key witnesses are the same as branches')
    for k, v in sorted(kvpairs):
        assert t.get_prefix_witness(k) == t.get_long_format_branch(k)
    print('Checking byte-wide witnesses')
    for _ in range(16):
        byte = random.randrange(256)
        witness = t.get_prefix_witness(bytearray([byte]))
        subtrie = Trie(EphemDB({sha3(x): x for x in witness}), t.root)
        print('auditing byte', byte, 'with', len([k for k,v in kvpairs if k[0] == byte]), 'keys')
        for k, v in sorted(kvpairs):
            if k[0] == byte:
                assert subtrie.get(k) == v
            assert subtrie.get(bytearray([byte] + [0] * 19)) == None
            assert subtrie.get(bytearray([byte] + [255] * 19)) == None
    for k, v in shuffle_in_place(kvpairs):
        t.update(k, b'')
        if not random.randrange(100):
            t.to_dict()
    #t.print_nodes()
    assert t.root == b''
