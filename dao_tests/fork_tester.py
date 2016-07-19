from ethereum.state import State
import json
from ethereum import abi
from ethereum.utils import normalize_address
from ethereum.state_transition import apply_transaction, apply_const_message
from ethereum.vm import Message, CallData
from ethereum.config import Env
from ethereum.parse_genesis_declaration import mk_basic_state
from ethereum.transactions import Transaction

account_dict = json.load(open('dao_dump.json'))
withdrawer_code = '0x' + open('bytecode.txt').read().strip()
true, false = True, False
withdrawer_ct = abi.ContractTranslator([{"constant":false,"inputs":[],"name":"trusteeWithdraw","outputs":[],"type":"function"},{"constant":false,"inputs":[],"name":"withdraw","outputs":[],"type":"function"},{"constant":true,"inputs":[],"name":"mainDAO","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":true,"inputs":[],"name":"trustee","outputs":[{"name":"","type":"address"}],"type":"function"}])
dao_ct = abi.ContractTranslator([{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_amount","type":"uint256"}],"name":"approve","outputs":[{"name":"success","type":"bool"}],"type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],"name":"transferFrom","outputs":[{"name":"success","type":"bool"}],"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},{"constant":true,"inputs":[],"name":"standard","outputs":[{"name":"","type":"string"}],"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_amount","type":"uint256"}],"name":"transfer","outputs":[{"name":"success","type":"bool"}],"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"remaining","type":"uint256"}],"type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_from","type":"address"},{"indexed":true,"name":"_to","type":"address"},{"indexed":false,"name":"_amount","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_owner","type":"address"},{"indexed":true,"name":"_spender","type":"address"},{"indexed":false,"name":"_amount","type":"uint256"}],"name":"Approval","type":"event"}])

# Initialize state

dao = "0xbb9bc244d798123fde783fcc1c72d3bb8c189413"
withdrawer = "0xbf4ed7b27f1d666546e30d74d50d173d20bca754"
my_account = "0x1db3439a222c519ab44bb1144fc28167b4fa6ee6"
my_other_account = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
curator = "0xda4a4626d3e16e094de3225a751aab7128e96526"

state = mk_basic_state({
    dao: account_dict,
    withdrawer: {
        "code": withdrawer_code,
        "balance": "12072877497524777000000000",
        "storage": {
            "0x": "0xda4a4626d3e16e094de3225a751aab7128e96526"
        }
    },
}, {
    "number": 1920001,
    "gas_limit": 4712388,
    "gas_used": 0,
    "timestamp": 1467446877,
    "difficulty": 2**25,
    "hash": '00' * 32,
    "uncles_hash": '00' * 32
}, Env())

def get_dao_balance(state, address):
    msg_data = CallData([ord(x) for x in dao_ct.encode('balanceOf', [address])])
    msg = Message(normalize_address(address), normalize_address(dao), 0, 1000000, msg_data, code_address=normalize_address(dao))
    output = ''.join(map(chr, apply_const_message(state, msg)))
    return dao_ct.decode('balanceOf', output)[0]

import sys
state.log_listeners.append(lambda x: sys.stdout.write(str(dao_ct.listen(x))+'\n'))
state.log_listeners.append(lambda x: sys.stdout.write(str(withdrawer_ct.listen(x))+'\n'))

print 'State created'

# Check pre-balance

pre_balance = state.get_balance(my_account)
pre_dao_tokens = get_dao_balance(state, my_account)
pre_withdrawer_balance = state.get_balance(withdrawer)

print 'Pre ETH (wei) balance: %d' % pre_balance
print 'Pre DAO (base unit) balance: %d' % pre_dao_tokens

# Attempt to claim the ETH without approving (should fail)

tx0 = Transaction(state.get_nonce(my_account), 0, 1000000, withdrawer, 0, withdrawer_ct.encode('withdraw', [])).sign('\x33' * 32)
tx0._sender = normalize_address(my_account)
apply_transaction(state, tx0)

med_balance = state.get_balance(my_account)
med_dao_tokens = get_dao_balance(state, my_account)
med_withdrawer_balance = state.get_balance(withdrawer)

assert med_balance == pre_balance
assert med_dao_tokens == pre_dao_tokens
assert med_withdrawer_balance == pre_withdrawer_balance > 0

print 'ETH claim without approving failed, as expected'

# Approve the withdrawal

tx1 = Transaction(state.get_nonce(my_account), 0, 1000000, dao, 0, dao_ct.encode('approve', [withdrawer, 100000 * 10**18])).sign('\x33' * 32)
tx1._sender = normalize_address(my_account)
apply_transaction(state, tx1)

# Check allowance

allowance = dao_ct.decode('allowance', ''.join(map(chr, apply_const_message(state, Message(normalize_address(my_account), normalize_address(dao), 0, 1000000, CallData([ord(x) for x in dao_ct.encode('allowance', [my_account, withdrawer])]), code_address=dao)))))[0]
assert allowance == 100000 * 10**18, allowance
print 'Allowance verified'

# Claim the ETH

tx2 = Transaction(state.get_nonce(my_account), 0, 1000000, withdrawer, 0, withdrawer_ct.encode('withdraw', [])).sign('\x33' * 32)
tx2._sender = normalize_address(my_account)
apply_transaction(state, tx2)

# Compare post_balance

post_balance = state.get_balance(my_account)
post_dao_tokens = get_dao_balance(state, my_account)

print 'Post ETH (wei) balance: %d' % post_balance
print 'Post DAO (base unit) balance: %d' % post_dao_tokens

assert post_dao_tokens == 0
assert post_balance - pre_balance == pre_dao_tokens

print 'Withdrawing once works'

# Try to claim post_balance again, should have no effect

tx3 = Transaction(state.get_nonce(my_account), 0, 1000000, withdrawer, 0, withdrawer_ct.encode('withdraw', [])).sign('\x33' * 32)
tx3._sender = normalize_address(my_account)
apply_transaction(state, tx3)

post_balance2 = state.get_balance(my_account)
post_dao_tokens2 = get_dao_balance(state, my_account)

assert post_balance2 == post_balance
assert post_dao_tokens2 == post_dao_tokens

# Curator withdraw

pre_curator_balance = state.get_balance(curator)
pre_withdrawer_balance = state.get_balance(withdrawer)

# from ethereum.slogging import LogRecorder, configure_logging, set_level
# config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
# configure_logging(config_string=config_string)

tx4 = Transaction(0, 0, 1000000, withdrawer, 0, withdrawer_ct.encode('trusteeWithdraw', [])).sign('\x33' * 32)
apply_transaction(state, tx4)

post_curator_balance = state.get_balance(curator)
post_withdrawer_balance = state.get_balance(withdrawer)
print 'Curator withdrawn', post_curator_balance - pre_curator_balance

assert 500000 * 10**18 < post_curator_balance - pre_curator_balance < 600000 * 10**18
assert pre_curator_balance + pre_withdrawer_balance == post_curator_balance + post_withdrawer_balance

tx5 = Transaction(1, 0, 1000000, withdrawer, 0, withdrawer_ct.encode('trusteeWithdraw', [])).sign('\x33' * 32)
apply_transaction(state, tx5)

post_curator_balance2 = state.get_balance(curator)
post_withdrawer_balance2 = state.get_balance(withdrawer)
assert post_curator_balance2 == post_curator_balance
assert post_withdrawer_balance2 == post_withdrawer_balance

print 'Second withdrawal has no effect as expected'

# Withdraw again, and try curator withdrawing again

tx6 = Transaction(state.get_nonce(my_other_account), 0, 1000000, dao, 0, dao_ct.encode('approve', [withdrawer, 100000 * 10**18])).sign('\x33' * 32)
tx6._sender = normalize_address(my_other_account)
apply_transaction(state, tx6)

tx7 = Transaction(state.get_nonce(my_other_account), 0, 1000000, withdrawer, 0, withdrawer_ct.encode('withdraw', [])).sign('\x33' * 32)
tx7._sender = normalize_address(my_other_account)
apply_transaction(state, tx7)

post_withdrawer_balance3 = state.get_balance(withdrawer)
print 'Another %d wei withdrawn' % (post_withdrawer_balance2 - post_withdrawer_balance3)
assert post_withdrawer_balance3 < post_withdrawer_balance2

tx8 = Transaction(2, 0, 1000000, withdrawer, 0, withdrawer_ct.encode('trusteeWithdraw', [])).sign('\x33' * 32)
apply_transaction(state, tx8)

post_curator_balance3 = state.get_balance(curator)
assert post_curator_balance2 == post_curator_balance

print 'Third withdrawal has no effect as expected'

# Withdraw from an account with no DAO

no_dao_account = '\x35' * 20

pre_balance = state.get_balance(no_dao_account)
pre_dao_tokens = get_dao_balance(state, no_dao_account)

tx9 = Transaction(state.get_nonce(no_dao_account), 0, 1000000, dao, 0, dao_ct.encode('approve', [withdrawer, 100000 * 10**18])).sign('\x33' * 32)
tx9._sender = no_dao_account
apply_transaction(state, tx9)

tx10 = Transaction(state.get_nonce(no_dao_account), 0, 1000000, withdrawer, 0, withdrawer_ct.encode('withdraw', [])).sign('\x33' * 32)
tx10._sender = no_dao_account
apply_transaction(state, tx10)

post_balance = state.get_balance(no_dao_account)
post_dao_tokens = get_dao_balance(state, no_dao_account)

assert pre_balance == post_balance == 0
assert pre_dao_tokens == post_dao_tokens

print 'Withdrawal from a non-DAO-holding account has no effect'
