
import serpent
import rlp
from ethereum import utils
from ethereum import tester
from ethereum import transactions

sighash = serpent.compile('sighash.se.py')

tests = [
    [b"\x01"],
    [b"\x80", "a"],
    [b"\x81", "b"],
    [b""],
    [b"", b"\x01", b""],
    [b"", b"\x81", b""],
    [b"dog", b"c" * 54, b"\x01"],
    [b"\x01", b"c" * 55, b"pig"],
    [b"moose", b"c" * 56, b"\x00"],
    [b'\x01', b'55555555555555555555555555555555', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x88\xa7\x85r\x1b3\x17\xcaP\x96\xca\xd3S\xfcgM\xec\xe0\xf5!\xc8\xb4m\xd9\xb7E\xf3\x81d\x87\x93VD\xe0Ej\xcd\xec\x80\x11\x86(qZ\x9b\x80\xbf\xce\xe5*\r\x9d.o\xcd\x11s\xc5\xbc\x8c\xcb\xb9\xa9 ']
]

s = tester.state()
c = s.evm(sighash, sender=tester.k0, endowment=0)

for test in tests:
    z = s.send(tester.k0, c, 0, rlp.encode(test)) 
    assert z == utils.sha3(rlp.encode(test[:-1]))
    print("Passed test, gas consumed: ", s.state.receipts[-1].gas_used - s.state.receipts[-2].gas_used - s.last_tx.intrinsic_gas_used)

# Create transaction
t = transactions.Transaction(0, 30 * 10**9, 2999999, '', 0, sighash)
t.startgas = t.intrinsic_gas_used + 50000 + 200 * len(sighash)
t.v = 27
t.r = 45
t.s = 79
print("Sighash")
print('Send %d wei to %s' % (t.startgas * t.gasprice,
                             '0x'+utils.encode_hex(t.sender)))

print('Contract address: 0x'+utils.encode_hex(utils.mk_contract_address(t.sender, 0)))
print('Code: 0x'+utils.encode_hex(rlp.encode(t)))
