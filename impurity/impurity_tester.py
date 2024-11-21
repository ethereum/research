from ethereum import tester as t
from ethereum import utils
from ethereum import transactions
import rlp
import serpent
s = t.state()
c = s.abi_contract('check_for_impurity.se')

#from ethereum.slogging import LogRecorder, configure_logging, set_level
#config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
#configure_logging(config_string=config_string)

test1 = s.abi_contract("""

data horse

def foo():
    return self.horse

""")

try:
    c.submit(test1.address)
    success = True
except:
    success = False
assert not success

failedtest_addr = "0x"+utils.encode_hex(test1.address)

test2 = s.abi_contract("""

def foo():
    return block.number

""")

try:
    c.submit(test2.address)
    success = True
except:
    success = False
assert not success

test3 = s.abi_contract("""

def foo(x):
    return x * 2
""")

c.submit(test3.address)



test4 = s.abi_contract("""

def modexp(b: uint256, e: uint256, m: uint256):
    if e == 0:
        return 1
    elif e == 1:
        return b
    elif e % 2 == 0:
        return self.modexp(~mulmod(b, b, m), ~div(e, 2), m)
    elif e % 2 == 1:
        return ~mulmod(self.modexp(~mulmod(b, b, m), ~div(e, 2), m), b, m)

""")

c.submit(test4.address)
modexp_addr = "0x"+utils.encode_hex(test4.address)

test5 = s.abi_contract("""

def modinv(b, m):
    inpdata = [0xa7d4bbe6, b, m-2, m]
    outdata = [0]
    ~call(100000, %s, 0, inpdata + 28, 100, outdata, 32)
    return outdata[0]
    
""" % modexp_addr)

c.submit(test5.address)

test6 = s.abi_contract("""
def phooey(h, v, r, s):
    return ecrecover(h, v, r, s)
""")

c.submit(test6.address)

test7 = s.abi_contract("""

def modinv(b, m):
    inpdata = [0xa7d4bbe6, b, m-2, m]
    outdata = [0]
    ~call(msg.gas - 10000, %s, 0, inpdata + 28, 100, outdata, 32)
    return outdata[0]
    
""" % failedtest_addr)

try:
    c.submit(test7.address)
    success = True
except:
    success = False
assert not success

print('All tests passed')

kode = serpent.compile('check_for_impurity.se')

# Create transaction
t = transactions.Transaction(0, 30 * 10**9, 2999999, '', 0, kode)
t.startgas = t.intrinsic_gas_used + 50000 + 200 * len(kode)
t.v = 27
t.r = 45
t.s = 79
print('Send %d wei to %s' % (t.startgas * t.gasprice,
                             '0x'+utils.encode_hex(t.sender)))

print('Contract address: 0x'+utils.encode_hex(utils.mk_contract_address(t.sender, 0)))
print('Code: 0x'+utils.encode_hex(rlp.encode(t)))
print('ABI declaration: '+repr(serpent.mk_full_signature('check_for_impurity.se')))
