from new_bintrie import Trie, EphemDB, encode_bin, encode_bin_path, decode_bin_path
from ethereum.utils import sha3, encode_hex
import random

def shuffle_in_place(x):
    y = x[::]
    random.shuffle(y)
    return y

kvpairs = [(sha3(str(i))[12:], str(i) * 5) for i in range(1000)]


for path in ([], [1,0,1], [0,0,1,0], [1,0,0,1,0], [1,0,0,1,0,0,1,0], [1,0] * 8):
    assert decode_bin_path(encode_bin_path(bytes(path))) == bytes(path)

r1 = None

for i in range(3):
    t = Trie(EphemDB(), b'')
    for k, v in shuffle_in_place(kvpairs):
        #print(t.to_dict())
        t.update(k, v)
        assert t.get(k) == v
        #t.print_nodes()
    assert r1 is None or t.root == r1
    r1 = t.root
    t.update(kvpairs[0][0], kvpairs[0][1])
    assert t.root == r1
    print(t.get_branch(kvpairs[0][0]))
    print(encode_hex(t.root))

t.to_dict()
