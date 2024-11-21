from ethereum.tools import tester as t
from ethereum import utils as u
c = t.Chain()
x = c.contract(open('utxos.v.py').read(), language='viper', sender=t.k0)
assert u.normalize_address(x.get_utxos__owner(u.encode_int32(1))) == t.a0
assert x.get_utxos__value(u.encode_int32(1)) == 2**32

sigdata = u.encode_int32(1) + u.encode_int32(0) + b'\x00' * 12 + t.a2 + \
    u.encode_int32(2**30) + b'\x00' * 12 + t.a2 + u.encode_int32(3 * 2**30)
sighash = u.sha3(sigdata)

v, r, s = u.ecsign(sighash, t.k0)

assert x.tx(u.encode_int32(1), u.encode_int32(0), t.a2, 2**30, t.a2, 3 * 2**30, v, r, s) == sighash
assert u.normalize_address(x.get_utxos__owner(sighash)) == t.a2
assert x.get_utxos__value(sighash) == 2**30

sigdata2 = sighash + u.encode_int32(0) + b'\x00' * 12 + t.a3 + \
    u.encode_int32(2**29) + b'\x00' * 12 + t.a3 + u.encode_int32(2**29)
sighash2 = u.sha3(sigdata2)

v, r, s = u.ecsign(sighash2, t.k2)

assert x.tx(sighash, u.encode_int32(0), t.a3, 2**29, t.a3, 2**29, v, r, s) == sighash2

print('Tests passed')
