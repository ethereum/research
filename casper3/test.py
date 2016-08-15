import networksim
from casper import Validator
import casper
from ethereum.parse_genesis_declaration import mk_basic_state
from ethereum.config import Env
from ethereum.casper_utils import RandaoManager, generate_validation_code, call_casper, \
    get_skips_and_block_making_time, sign_block, make_block, get_contract_code, \
    casper_config, get_casper_ct, get_casper_code, get_rlp_decoder_code, \
    get_hash_without_ed_code, make_casper_genesis, validator_sizes, find_indices
from ethereum.utils import sha3, privtoaddr
from ethereum.transactions import Transaction
from ethereum.state_transition import apply_transaction

from ethereum.slogging import LogRecorder, configure_logging, set_level
# config_string = ':info,eth.vm.log:trace,eth.vm.op:trace,eth.vm.stack:trace,eth.vm.exit:trace,eth.pb.msg:trace,eth.pb.tx:debug'
config_string = ':info,eth.vm.log:trace'
configure_logging(config_string=config_string)

n = networksim.NetworkSimulator(latency=150)
n.time = 2
print 'Generating keys'
keys = [sha3(str(i)) for i in range(20)]
print 'Initializing randaos'
randaos = [RandaoManager(sha3(k)) for k in keys]
deposit_sizes = [128] * 15 + [256] * 5

print 'Creating genesis state'
s = make_casper_genesis(validators=[(generate_validation_code(privtoaddr(k)), ds * 10**18, r.get(9999))
                                    for k, ds, r in zip(keys, deposit_sizes, randaos)],
                        alloc={privtoaddr(k): {'balance': 10**18} for k in keys},
                        timestamp=2,
                        epoch_length=40)
g = s.to_snapshot()
print 'Genesis state created'

validators = [Validator(g, k, n, Env(config=casper_config), time_offset=4) for k in keys]
n.agents = validators
n.generate_peers()
lowest_shared_height = -1
made_101_check = 0

for i in range(100000):
    # print 'ticking'
    n.tick()
    if i % 100 == 0:
        print '%d ticks passed' % i
        print 'Validator heads:', [v.chain.head.header.number if v.chain.head else None for v in validators]
        print 'Total blocks created:', casper.global_block_counter
        print 'Dunkle count:', call_casper(validators[0].chain.state, 'getTotalDunklesIncluded', [])
        lowest_shared_height = min([v.chain.head.header.number if v.chain.head else -1 for v in validators])
        if lowest_shared_height >= 101 and not made_101_check:
            made_101_check = True
            print 'Checking that withdrawn validators are inactive'
            assert len([v for v in validators if v.active]) == len(validators) - 5, len([v for v in validators if v.active])
            print 'Check successful'
            break
    if i == 1:
        print 'Checking that all validators are active'
        assert len([v for v in validators if v.active]) == len(validators)
        print 'Check successful'
    if i == 2000:
        print 'Withdrawing a few validators'
        for v in validators[:5]:
            v.withdraw()
    if i == 4000:
        print 'Checking that validators have withdrawn'
        for v in validators[:5]:
            assert call_casper(v.chain.state, 'getEndEpoch', []) <= 2
        print 'Check successful'
