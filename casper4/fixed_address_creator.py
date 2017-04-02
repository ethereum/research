from ethereum import transactions, utils
import serpent
import rlp

sighash = serpent.compile('sighash.se.py')

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

sighash = serpent.compile('sqrt.se.py')

# Create transaction
t = transactions.Transaction(0, 30 * 10**9, 2999999, '', 0, sighash)
t.startgas = t.intrinsic_gas_used + 50000 + 200 * len(sighash)
t.v = 27
t.r = 45
t.s = 79
print("Sqrt")
print('Send %d wei to %s' % (t.startgas * t.gasprice,
                             '0x'+utils.encode_hex(t.sender)))

print('Contract address: 0x'+utils.encode_hex(utils.mk_contract_address(t.sender, 0)))
print('Code: 0x'+utils.encode_hex(rlp.encode(t)))
