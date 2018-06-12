from bls import G1, G2, hash_to_G2, compress_G1, compress_G2, \
    decompress_G1, decompress_G2, normalize, multiply, \
    sign, privtopub, aggregate_sigs, aggregate_pubs, verify

from simpleserialize import serialize, deserialize, eq

from full_pos import ActiveState, CheckpointRecord

for x in (1, 5, 124, 735, 127409812145, 90768492698215092512159, 0):
    print('Testing with privkey %d' % x)
    p1 = multiply(G1, x)
    p2 = multiply(G2, x)
    msg = str(x).encode('utf-8')
    msghash = hash_to_G2(msg)
    assert normalize(decompress_G1(compress_G1(p1))) == normalize(p1)
    assert normalize(decompress_G2(compress_G2(p2))) == normalize(p2)
    assert normalize(decompress_G2(compress_G2(msghash))) == normalize(msghash)
    sig = sign(msg, x)
    pub = privtopub(x)
    assert verify(msg, pub, sig)

print('Testing signature aggregation')
msg = b'cow'
keys = [1, 5, 124, 735, 127409812145, 90768492698215092512159, 0]
sigs = [sign(msg, k) for k in keys]
pubs = [privtopub(k) for k in keys]
aggsig = aggregate_sigs(sigs)
aggpub = aggregate_pubs(pubs)
assert verify(msg, aggpub, aggsig)

print('Testing basic serialization')

assert serialize(5, 'int8') == b'\x05'
assert deserialize(b'\x05', 'int8') == 5
assert serialize(2**32-3, 'int40') == b'\x00\xff\xff\xff\xfd'
assert deserialize(b'\x00\xff\xff\xff\xfd', 'int40') == 2**32-3
assert serialize(b'\x35'*20, 'address') == b'\x35'*20
assert deserialize(b'\x35'*20, 'address') == b'\x35'*20
assert serialize(b'\x35'*32, 'hash32') == b'\x35'*32
assert deserialize(b'\x35'*32, 'hash32') == b'\x35'*32
assert serialize(b'cow', 'bytes') == b'\x00\x00\x00\x03cow'
assert deserialize(b'\x00\x00\x00\x03cow', 'bytes') == b'cow'

print('Testing advanced serialization')


s = ActiveState()
ds = deserialize(serialize(s, type(s)), type(s))
assert eq(s, ds)
s = ActiveState(checkpoints=[CheckpointRecord(checkpoint_hash=b'\x55'*32, bitmask=b'31337dawg')],
                height=555, randao=b'\x88'*32, balance_deltas=[5,7,9,579] + [3] * 333)
ds = deserialize(serialize(s, type(s)), type(s))
assert eq(s, ds)
