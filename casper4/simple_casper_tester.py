from ethereum import tester as t
from ethereum import utils, state_transition, transactions
from viper import compiler
#from ethereum.slogging import LogRecorder, configure_logging, set_level
#config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
#configure_logging(config_string=config_string)
import rlp
s = t.state()
t.languages['viper'] = compiler.Compiler()
t.gas_limit = 9999999

EPOCH_LENGTH = 256

# Install RLP decoder library
s.state.set_balance('0xfe2ec957647679d210034b65e9c7db2452910b0c', 9350880000000000)
state_transition.apply_transaction(s.state, rlp.decode(utils.decode_hex('f903bd808506fc23ac008304c1908080b903aa6103988061000e6000396103a65660006101bf5361202059905901600090526101008152602081019050602052600060605261040036018060200159905901600090528181526020810190509050608052600060e0527f0100000000000000000000000000000000000000000000000000000000000000600035046101005260c061010051121561007e57fe5b60f86101005112156100a95760c061010051036001013614151561009e57fe5b6001610120526100ec565b60f761010051036020036101000a600161012051013504610140526101405160f7610100510360010101361415156100dd57fe5b60f76101005103600101610120525b5b366101205112156102ec577f01000000000000000000000000000000000000000000000000000000000000006101205135046101005260e0516060516020026020510152600160605101606052608061010051121561017a57600160e0516080510152600161012051602060e0516080510101376001610120510161012052602160e0510160e0526102da565b60b8610100511215610218576080610100510360e05160805101526080610100510360016101205101602060e05160805101013760816101005114156101ef5760807f010000000000000000000000000000000000000000000000000000000000000060016101205101350412156101ee57fe5b5b600160806101005103016101205101610120526020608061010051030160e0510160e0526102d9565b60c06101005112156102d65760b761010051036020036101000a6001610120510135046101405260007f0100000000000000000000000000000000000000000000000000000000000000600161012051013504141561027357fe5b603861014051121561028157fe5b6101405160e05160805101526101405160b761010051600161012051010103602060e05160805101013761014051600160b7610100510301016101205101610120526020610140510160e0510160e0526102d8565bfe5b5b5b602060605113156102e757fe5b6100ed565b60e051606051602002602051015261082059905901600090526108008152602081019050610160526000610120525b6060516101205113151561035c576020602060605102610120516020026020510151010161012051602002610160510152600161012051016101205261031b565b60e0518060206020606051026101605101018260805160006004600a8705601201f161038457fe5b50602060e051602060605102010161016051f35b6000f31b2d4f'), transactions.Transaction))
assert s.state.get_code('0x0b8178879f97f2ada01fb8d219ee3d0ad74e91e0')

# Install sig hasher

sighasher = s.contract(open('sighash.se.py').read(), language='serpent')


casper_code = open('simple_casper.v.py').read().replace('0x1db3439a222c519ab44bb1144fc28167b4fa6ee6', '0x'+utils.encode_hex(t.a0)) \
                                               .replace('0x38920146f10f3956fc09970beededcb2d9638712', '0x'+utils.encode_hex(sighasher))

print(casper_code)
print(utils.encode_hex(sighasher))

print(len(compiler.compile(casper_code)))

casper = s.abi_contract(casper_code, language='viper', startgas=5555555)

def mk_prepare(epoch, hash, ancestry_hash, source_epoch, source_ancestry_hash, key):
    sighash = utils.sha3(rlp.encode([epoch, hash, ancestry_hash, source_epoch, source_ancestry_hash]))
    v, r, s = utils.ecdsa_raw_sign(sighash, key)
    sig = utils.encode_int32(v) + utils.encode_int32(r) + utils.encode_int32(s)
    return rlp.encode([epoch, hash, ancestry_hash, source_epoch, source_ancestry_hash, sig])

def mk_commit(epoch, hash, key):
    sighash = utils.sha3(rlp.encode([epoch, hash]))
    v, r, s = utils.ecdsa_raw_sign(sighash, key)
    sig = utils.encode_int32(v) + utils.encode_int32(r) + utils.encode_int32(s)
    return rlp.encode([epoch, hash, sig])

s.state.block_number = EPOCH_LENGTH

# Initialize the first epoch
casper.initialize_epoch(1)
assert casper.get_nextValidatorIndex() == 1
# Send a prepare message
casper.prepare(0, mk_prepare(1, '\x35' * 32, '\x00' * 32, 0, '\x00' * 32, t.k0))
epoch_1_anchash = utils.sha3(b'\x35' * 32 + b'\x00' * 32)
assert casper.get_consensus_messages__hash_justified(1, b'\x35' * 32)
assert casper.get_consensus_messages__ancestry_hash_justified(1, epoch_1_anchash)
# Send a commit message
casper.commit(0, mk_commit(1, '\x35' * 32, t.k0))
# Check that we committed
assert casper.get_consensus_messages__committed(1)
# Initialize the second epoch 
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(2)
# Check that the dynasty increased as expected
assert casper.get_dynasty() == 1
assert casper.get_total_deposits(1) == casper.get_total_deposits(0) > 0
# Send a prepare message
casper.prepare(0, mk_prepare(2, '\x45' * 32, epoch_1_anchash, 1, epoch_1_anchash, t.k0))
# Send a commit message
epoch_2_commit = mk_commit(2, '\x45' * 32, t.k0)
casper.commit(0, epoch_2_commit)
epoch_2_anchash = utils.sha3(b'\x45' * 32 + epoch_1_anchash)
assert casper.get_consensus_messages__ancestry_hash_justified(2, epoch_2_anchash)
# Check that we committed
assert casper.get_consensus_messages__committed(2)
# Initialize the third epoch
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(3)
# Test the NO_DBL_PREPARE slashing condition
p1 = mk_prepare(3, '\x56' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k0)
p2 = mk_prepare(3, '\x57' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k0)
snapshot = s.snapshot()
casper.double_prepare_slash(0, p1, p2)
s.revert(snapshot)
# Test the PREPARE_COMMIT_CONSISTENCY slashing condition
p3 = mk_prepare(3, '\x58' * 32, epoch_2_anchash, 0, b'\x00' * 32, t.k0)
snapshot = s.snapshot()
casper.prepare_commit_inconsistency_slash(0, p3, epoch_2_commit)
s.revert(snapshot)
# Finish the third epoch
casper.prepare(0, p1)
casper.commit(0, mk_commit(3, '\x56' * 32, t.k0))
epoch_3_anchash = utils.sha3(b'\x56' * 32 + epoch_2_anchash)
assert casper.get_consensus_messages__ancestry_hash_justified(3, epoch_3_anchash)
assert casper.get_consensus_messages__committed(3)
# Initialize the fourth epoch. Not doing prepares or commits during this epoch.
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(4)
assert casper.get_dynasty() == 3
epoch_4_anchash = utils.sha3(b'\x67' * 32 + epoch_3_anchash)
# Not publishing this prepare for the time being
p4 = mk_prepare(4, '\x78' * 32, '\x12' * 32, 3, '\x24' * 32, t.k0)
# Initialize the fifth epoch
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(5)
# Dynasty not incremented because no commits were made
assert casper.get_dynasty() == 3
epoch_5_anchash = utils.sha3(b'\x78' * 32 + epoch_4_anchash)
p5 = mk_prepare(5, '\x78' * 32, epoch_4_anchash, 3, epoch_3_anchash, t.k0)
casper.prepare(0, p5)
# Test the COMMIT_REQ slashing condition
kommit = mk_commit(5, b'\x80' * 32, t.k0)
s.state.block_number += EPOCH_LENGTH * 30
for i in range(6, 36):
    casper.initialize_epoch(i)
snapshot = s.snapshot()
casper.commit_non_justification_slash(0, kommit)
s.revert(snapshot)
try:
    casper.commit_non_justification_slash(0, epoch_2_commit)
    success = True
except:
    success = False
assert not success
# Test the PREPARE_REQ slashing condition
casper.derive_ancestry(epoch_3_anchash, epoch_2_anchash, epoch_1_anchash)
snapshot = s.snapshot()
casper.prepare_non_justification_slash(0, p4)
s.revert(snapshot)
try:
    casper.prepare_non_justification_slash(0, p5)
    success = True
except:
    success = False
assert not success
