from ethereum import tester, vm
from ethereum.utils import sha3, encode_int32, safe_ord, encode_hex
from ethereum.state_transition import apply_message
s = tester.state()
c = s.contract('eip_96_blockhash_getter.se.py')
blockhash_addr = b'\x00' * 19 + b'\x10'
system_addr = b'\xff' * 19 + b'\xfe'
s.state.set_code(blockhash_addr, s.state.get_code(c))

def mk_hash_setting_message(data):
    return vm.Message(sender=system_addr, to=blockhash_addr, value=0, gas=1000000, data=data)

print("Setting block hashes")
for i in range(1, 1000):
    s.state.block_number = i + 1
    o = apply_message(s.state, mk_hash_setting_message(sha3(str(i))))
    if i % 100 == 0:
        print("Set %d" % i)

print("Testing reads")
s.state.block_number = 1000
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(999)) == sha3(str(999))
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(998)) == sha3(str(998))
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(744)) == sha3(str(744))
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(743)) == b'\x00' * 32
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(1000)) == b'\x00' * 32
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(1001)) == b'\x00' * 32
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(513)) == b'\x00' * 32
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(512)) == sha3(str(512))
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(511)) == b'\x00' * 32
assert s.send(tester.k0, blockhash_addr, 0, encode_int32(256)) == sha3(str(256))
print("Tests passed!")

print("EVM code: 0x%s" % encode_hex(s.state.get_code(blockhash_addr)))
