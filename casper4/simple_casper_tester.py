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

s.state.set_balance('0x6e7406512b244843c1171840dfcd3d7532d979fe', 7291200000000000)

state_transition.apply_transaction(s.state, rlp.decode(utils.decode_hex('f902b9808506fc23ac008303b5608080b902a66102948061000e6000396102a2567f01000000000000000000000000000000000000000000000000000000000000006000350460205260c0602051121561003857fe6100a7565b60f8602051121561005657600160405260c0602051036060526100a6565b60f76020510360010160405260007f010000000000000000000000000000000000000000000000000000000000000060013504141561009157fe5b60f7602051036020036101000a600135046060525b5b36606051604051011415156100b857fe5b604051608052600060a0525b3660405112156101c0577f0100000000000000000000000000000000000000000000000000000000000000604051350460c052608060c05112156101165760405160a0526001604051016040526101bb565b60b860c051121561014257608060c0510360605260405160a052600160605101604051016040526101ba565b60c060c05112156101b75760007f01000000000000000000000000000000000000000000000000000000000000006001604051013504141561018057fe5b60b760c051036020036101000a600160405101350460605260405160a052600160b760c051036060510101604051016040526101b9565bfe5b5b5b6100c4565b60805160a0510360e0526103e861010052603860e051121561020f5760e05160c001610100515360e051608051600161010051013760e0516001016101005120610120526020610120f3610293565b60006101405260e051610160525b610160511561024257600161014051016101405261010061016051046101605261021d565b6101405160f7016101005153610140516020036101000a60e05102600161010051015260e0516080516101405160016101005101013760e05161014051600101016101005120610180526020610180f35b5b6000f31b2d4f'), transactions.Transaction))
assert s.state.get_code('0x476c2ca9a7f3b16feca86512276271faf63b6a24')

# Install Casper

casper_code = open('simple_casper.v.py').read().replace('0x1db3439a222c519ab44bb1144fc28167b4fa6ee6', '0x'+utils.encode_hex(t.a0))

print('Casper code length', len(compiler.compile(casper_code)))

casper = s.abi_contract(casper_code, language='viper', startgas=5555555)

print('Gas consumed to launch Casper', s.state.receipts[-1].gas_used - s.state.receipts[-2].gas_used)

# Helper functions for making a prepare, commit, login and logout message

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

def mk_status_flicker(epoch, login, key):
    sighash = utils.sha3(rlp.encode([epoch, login]))
    v, r, s = utils.ecdsa_raw_sign(sighash, key)
    sig = utils.encode_int32(v) + utils.encode_int32(r) + utils.encode_int32(s)
    return rlp.encode([epoch, login, sig])

# Begin the test

print("Starting tests")
# Initialize the first epoch
s.state.block_number = EPOCH_LENGTH
casper.initialize_epoch(1)
assert casper.get_nextValidatorIndex() == 1
start = s.snapshot()
print("Epoch initialized")
# Send a prepare message
casper.prepare(0, mk_prepare(1, '\x35' * 32, '\x00' * 32, 0, '\x00' * 32, t.k0))
epoch_1_anchash = utils.sha3(b'\x35' * 32 + b'\x00' * 32)
assert casper.get_consensus_messages__hash_justified(1, b'\x35' * 32)
assert casper.get_consensus_messages__ancestry_hash_justified(1, epoch_1_anchash)
print("Prepare message processed")
# Send a commit message
casper.commit(0, mk_commit(1, '\x35' * 32, t.k0))
# Check that we committed
assert casper.get_consensus_messages__committed(1)
print("Commit message processed")
# Initialize the second epoch 
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(2)
# Check that the dynasty increased as expected
assert casper.get_dynasty() == 1
assert casper.get_total_deposits(1) == casper.get_total_deposits(0) > 0
print("Second epoch initialized, dynasty increased as expected")
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
print("Second epoch prepared and committed, third epoch initialized")
# Test the NO_DBL_PREPARE slashing condition
p1 = mk_prepare(3, '\x56' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k0)
p2 = mk_prepare(3, '\x57' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k0)
snapshot = s.snapshot()
casper.double_prepare_slash(0, p1, p2)
s.revert(snapshot)
print("NO_DBL_PREPARE slashing condition works")
# Test the PREPARE_COMMIT_CONSISTENCY slashing condition
p3 = mk_prepare(3, '\x58' * 32, epoch_2_anchash, 0, b'\x00' * 32, t.k0)
snapshot = s.snapshot()
casper.prepare_commit_inconsistency_slash(0, p3, epoch_2_commit)
s.revert(snapshot)
print("PREPARE_COMMIT_CONSISTENCY slashing condition works")
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
print("Epochs up to 5 initialized")
# Dynasty not incremented because no commits were made
assert casper.get_dynasty() == 3
epoch_5_anchash = utils.sha3(b'\x78' * 32 + epoch_4_anchash)
p5 = mk_prepare(5, '\x78' * 32, epoch_4_anchash, 3, epoch_3_anchash, t.k0)
casper.prepare(0, p5)
# Test the COMMIT_REQ slashing condition
kommit = mk_commit(5, b'\x80' * 32, t.k0)
s.state.block_number += EPOCH_LENGTH * 30
print("Speeding up time to test remaining two slashing conditions")
for i in range(6, 36):
    casper.initialize_epoch(i)
print("Epochs up to 36 initialized")
snapshot = s.snapshot()
casper.commit_non_justification_slash(0, kommit)
s.revert(snapshot)
try:
    casper.commit_non_justification_slash(0, epoch_2_commit)
    success = True
except:
    success = False
assert not success
print("COMMIT_REQ slashing condition works")
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
print("PREPARE_REQ slashing condition works")

print("Restarting the chain for test 2")
# Restart the chain
s.revert(start)
assert casper.get_dynasty() == 0
assert casper.get_current_epoch() == 1
assert casper.get_consensus_messages__ancestry_hash_justified(0, b'\x00' * 32)
print("Epoch 1 initialized")
for k in (t.k1, t.k2, t.k3, t.k4, t.k5, t.k6):
    casper.deposit(utils.privtoaddr(k), utils.privtoaddr(k), value=3 * 10**15)
print("Processed 6 deposits")
casper.prepare(0, mk_prepare(1, b'\x10' * 32, b'\x00' * 32, 0, b'\x00' * 32, t.k0))
casper.commit(0, mk_commit(1, b'\x10' * 32, t.k0))
epoch_1_anchash = utils.sha3(b'\x10' * 32 + b'\x00' * 32)
assert casper.get_consensus_messages__committed(1)
print("Prepared and committed")
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(2)
print("Epoch 2 initialized")
assert casper.get_dynasty() == 1
casper.prepare(0, mk_prepare(2, b'\x20' * 32, epoch_1_anchash, 1, epoch_1_anchash, t.k0))
casper.commit(0, mk_commit(2, b'\x20' * 32, t.k0))
epoch_2_anchash = utils.sha3(b'\x20' * 32 + epoch_1_anchash)
assert casper.get_consensus_messages__committed(2)
print("Confirmed that one key is still sufficient to prepare and commit")
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(3)
print("Epoch 3 initialized")
assert casper.get_dynasty() == 2
assert 3 * 10**15 <= casper.get_total_deposits(0) < 4 * 10**15
assert 3 * 10**15 <= casper.get_total_deposits(1) < 4 * 10**15
assert 21 * 10**15 <= casper.get_total_deposits(2) < 22 * 10**15
print("Confirmed new total_deposits")
try:
    # Try to log out, but sign with the wrong key
    casper.flick_status(0, mk_status_flicker(3, 0, t.k1))
    success = True
except:
    success = False
assert not success
# Log out
casper.flick_status(4, mk_status_flicker(3, 0, t.k4))
casper.flick_status(5, mk_status_flicker(3, 0, t.k5))
casper.flick_status(6, mk_status_flicker(3, 0, t.k6))
print("Logged out three validators")
# Validators leave the fwd validator set in dynasty 4
assert casper.get_validators__dynasty_end(4) == 4
epoch_3_anchash = utils.sha3(b'\x30' * 32 + epoch_2_anchash)
# Prepare from one validator
casper.prepare(0, mk_prepare(3, b'\x30' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k0))
# Not prepared yet
assert not casper.get_consensus_messages__hash_justified(3, b'\x30' * 32)
print("Prepare from one validator no longer sufficient")
# Prepare from 3 more validators
for i, k in ((1, t.k1), (2, t.k2), (3, t.k3)):
    casper.prepare(i, mk_prepare(3, b'\x30' * 32, epoch_2_anchash, 2, epoch_2_anchash, k))
# Still not prepared
assert not casper.get_consensus_messages__hash_justified(3, b'\x30' * 32)
print("Prepare from four of seven validators still not sufficient")
# Prepare from a firth validator
casper.prepare(4, mk_prepare(3, b'\x30' * 32, epoch_2_anchash, 2, epoch_2_anchash, t.k4))
# NOW we're prepared!
assert casper.get_consensus_messages__hash_justified(3, b'\x30' * 32)
print("Prepare from five of seven validators sufficient!")
# Five commits
for i, k in enumerate([t.k0, t.k1, t.k2, t.k3, t.k4]):
    casper.commit(i, mk_commit(3, b'\x30' * 32, k))
# And we committed!
assert casper.get_consensus_messages__committed(3)
print("Commit from five of seven validators sufficient")
# Start epoch 4
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(4)
assert casper.get_dynasty() == 3
print("Epoch 4 initialized")
# Prepare and commit
epoch_4_anchash = utils.sha3(b'\x40' * 32 + epoch_3_anchash)
for i, k in enumerate([t.k0, t.k1, t.k2, t.k3, t.k4]):
    casper.prepare(i, mk_prepare(4, b'\x40' * 32, epoch_3_anchash, 3, epoch_3_anchash, k))
for i, k in enumerate([t.k0, t.k1, t.k2, t.k3, t.k4]):
    casper.commit(i, mk_commit(4, b'\x40' * 32, k))
assert casper.get_consensus_messages__committed(4)
print("Prepared and committed")
# Start epoch 5 / dynasty 4
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(5)
print("Epoch 5 initialized")
assert casper.get_dynasty() == 4
assert 21 * 10**15 <= casper.get_total_deposits(3) <= 22 * 10**15
assert 12 * 10**15 <= casper.get_total_deposits(4) <= 13 * 10**15
epoch_5_anchash = utils.sha3(b'\x50' * 32 + epoch_4_anchash)
# Do three prepares
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.prepare(i, mk_prepare(5, b'\x50' * 32, epoch_4_anchash, 4, epoch_4_anchash, k))
# Three prepares are insufficient because there are still five validators in the rear validator set
assert not casper.get_consensus_messages__hash_justified(5, b'\x50' * 32)
print("Three prepares insufficient, as rear validator set still has seven")
# Do two more prepares
for i, k in [(3, t.k3), (4, t.k4)]:
    casper.prepare(i, mk_prepare(5, b'\x50' * 32, epoch_4_anchash, 4, epoch_4_anchash, k))
# Now we're good!
assert casper.get_consensus_messages__hash_justified(5, b'\x50' * 32)
print("Five prepares sufficient")
for i, k in enumerate([t.k0, t.k1, t.k2, t.k3, t.k4]):
    casper.commit(i, mk_commit(5, b'\x50' * 32, k))
# Committed!
assert casper.get_consensus_messages__committed(5)
# Start epoch 6 / dynasty 5
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(6)
assert casper.get_dynasty() == 5
print("Epoch 6 initialized")
# Log back in
casper.flick_status(4, mk_status_flicker(6, 1, t.k4))
print("One validator logging back in")
assert casper.get_validators__dynasty_start(4) == 7
# Here three prepares and three commits should be sufficient!
epoch_6_anchash = utils.sha3(b'\x60' * 32 + epoch_5_anchash)
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.prepare(i, mk_prepare(6, b'\x60' * 32, epoch_5_anchash, 5, epoch_5_anchash, k))
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.commit(i, mk_commit(6, b'\x60' * 32, k))
assert casper.get_consensus_messages__committed(6)
print("Three of four prepares and commits sufficient")
# Start epoch 7 / dynasty 6
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(7)
assert casper.get_dynasty() == 6
print("Epoch 7 initialized")
# Here three prepares and three commits should be sufficient!
epoch_7_anchash = utils.sha3(b'\x70' * 32 + epoch_6_anchash)
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.prepare(i, mk_prepare(7, b'\x70' * 32, epoch_6_anchash, 6, epoch_6_anchash, k))
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.commit(i, mk_commit(7, b'\x70' * 32, k))
assert casper.get_consensus_messages__committed(7)
print("Three of four prepares and commits sufficient")
# Start epoch 8 / dynasty 7
s.state.block_number += EPOCH_LENGTH
casper.initialize_epoch(8)
assert casper.get_dynasty() == 7
print("Epoch 8 initialized")
assert 12 * 10**15 <= casper.get_total_deposits(6) <= 13 * 10**15
assert 15 * 10**15 <= casper.get_total_deposits(7) <= 16 * 10**15
epoch_8_anchash = utils.sha3(b'\x80' * 32 + epoch_7_anchash)
# Do three prepares
for i, k in enumerate([t.k0, t.k1, t.k2]):
    casper.prepare(i, mk_prepare(8, b'\x80' * 32, epoch_7_anchash, 7, epoch_7_anchash, k))
# Three prepares are insufficient because there are still five validators in the rear validator set
assert not casper.get_consensus_messages__hash_justified(8, b'\x80' * 32)
print("Three prepares no longer sufficient, as the forward validator set has five validators")
# Do one more prepare
for i, k in [(3, t.k3)]:
    casper.prepare(i, mk_prepare(8, b'\x80' * 32, epoch_7_anchash, 7, epoch_7_anchash, k))
# Now we're good!
assert casper.get_consensus_messages__hash_justified(8, b'\x80' * 32)
print("Four of five prepares sufficient")
for i, k in enumerate([t.k0, t.k1, t.k2, t.k3, t.k4]):
    casper.commit(i, mk_commit(8, b'\x80' * 32, k))
assert casper.get_consensus_messages__committed(8)
print("Committed")

print("All tests passed")
