from ethereum.casper_utils import RandaoManager, get_skips_and_block_making_time, \
    generate_validation_code, call_casper, make_block, check_skips, get_timestamp, \
    get_casper_ct, validator_sizes
from ethereum.utils import sha3, hash32, privtoaddr, ecsign, zpad, encode_int32, \
    big_endian_to_int
from ethereum.block import Block
from ethereum.transactions import Transaction
from ethereum.chain import Chain
import networksim
import rlp
import random

CHECK_FOR_UNCLES_BACK = 8

global_block_counter = 0

casper_ct = get_casper_ct()

class ChildRequest(rlp.Serializable):
    fields = [
        ('prevhash', hash32)
    ]

    def __init__(self, prevhash):
        self.prevhash = prevhash

    @property
    def hash(self):
        return sha3(self.prevhash + '::salt:jhfqou213nry138o2r124124')

ids = []

class Validator():
    def __init__(self, genesis, key, network, env, time_offset=5):
        # Create a chain object
        self.chain = Chain(genesis, env=env)
        # Use the validator's time as the chain's time
        self.chain.time = lambda: self.get_timestamp()
        # My private key
        self.key = key
        # My address
        self.address = privtoaddr(key)
        # My randao
        self.randao = RandaoManager(sha3(self.key))
        # Pointer to the test p2p network
        self.network = network
        # Record of objects already received and processed
        self.received_objects = {}
        # The minimum eligible timestamp given a particular number of skips
        self.next_skip_count = 0
        self.next_skip_timestamp = 0
        # This validator's indices in the state
        self.indices = None
        # Is this validator active?
        self.active = False
        # Code that verifies signatures from this validator
        self.validation_code = generate_validation_code(privtoaddr(key))
        # Parents that this validator has already built a block on
        self.used_parents = {}
        # This validator's clock offset (for testing purposes)
        self.time_offset = random.randrange(time_offset) - (time_offset // 2)
        # Determine the epoch length
        self.epoch_length = self.call_casper('getEpochLength')
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)
        self.find_my_indices()
        self.cached_head = self.chain.head_hash

    def call_casper(self, fun, args=[]):
        return call_casper(self.chain.state, fun, args)

    def find_my_indices(self):
        epoch = self.chain.state.block_number // self.epoch_length
        print 'Finding indices for epoch %d' % epoch, self.call_casper('getEpoch')
        for i in range(len(validator_sizes)):
            valcount = self.call_casper('getHistoricalValidatorCount', [epoch, i])
            print i, valcount, self.call_casper('getHistoricalValidatorCount', [0, i])
            for j in range(valcount):
                valcode = self.call_casper('getValidationCode', [i, j])
                print (valcode, self.validation_code)
                if valcode == self.validation_code:
                    self.indices = i, j
                    start = self.call_casper('getStartEpoch', [i, j])
                    end = self.call_casper('getEndEpoch', [i, j])
                    if start <= epoch < end:
                        self.active = True
                        self.next_skip_count = 0
                        self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
                        print 'In current validator set at (%d, %d)' % (i, j)
                        return
                    else:
                        self.indices = None
                        self.active = False
                        self.next_skip_count, self.next_skip_timestamp = 0, 0
                        print 'Registered at (%d, %d) but not in current set' % (i, j)
                        return
        self.indices = None
        self.active = False
        self.next_skip_count, self.next_skip_timestamp = 0, 0
        print 'Not in current validator set'

    def get_uncles(self):
        anc = self.chain.get_block(self.chain.get_blockhash_by_number(self.chain.state.block_number - CHECK_FOR_UNCLES_BACK))
        if anc:
            descendants = self.chain.get_descendants(anc)
        else:
            descendants = self.chain.get_descendants(self.chain.db.get('GENESIS_HASH'))
        potential_uncles = [x for x in descendants if x not in self.chain and isinstance(x, Block)]
        uncles = [x.header for x in potential_uncles if not call_casper(self.chain.state, 'isDunkleIncluded', [x.header.hash])]
        return uncles

    def get_timestamp(self):
        return int(self.network.time * 0.01) + self.time_offset

    def on_receive(self, obj):
        if isinstance(obj, list):
            for _obj in obj:
                self.on_receive(_obj)
            return
        if obj.hash in self.received_objects:
            return
        if isinstance(obj, Block):
            print 'Receiving block', obj
            assert obj.hash not in self.chain
            block_success = self.chain.add_block(obj)
            self.network.broadcast(self, obj)
            self.network.broadcast(self, ChildRequest(obj.header.hash))
            self.update_head()
        elif isinstance(obj, Transaction):
            if self.chain.add_transaction(obj):
                self.network.broadcast(self, obj)
        self.received_objects[obj.hash] = True
        for x in self.chain.get_chain():
            assert x.hash in self.received_objects

    def tick(self):
        # Try to create a block
        # Conditions:
        # (i) you are an active validator,
        # (ii) you have not yet made a block with this parent
        if self.indices and self.chain.head_hash not in self.used_parents:
            t = self.get_timestamp()
            # Is it early enough to create the block?
            if t >= self.next_skip_timestamp and (not self.chain.head or t > self.chain.head.header.timestamp):
                # Wrong validator; in this case, just wait for the next skip count
                if not check_skips(self.chain, self.indices, self.next_skip_count):
                    self.next_skip_count += 1
                    self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
                    # print 'Incrementing proposed timestamp for block %d to %d' % \
                    #     (self.chain.head.header.number + 1 if self.chain.head else 0, self.next_skip_timestamp)
                    return
                self.used_parents[self.chain.head_hash] = True
                # Simulated 15% chance of validator failure to make a block
                if random.random() > 0.999:
                    print 'Simulating validator failure, block %d not created' % (self.chain.head.header.number + 1 if self.chain.head else 0)
                    return
                # Make the block, make sure it's valid
                pre_dunkle_count = call_casper(self.chain.state, 'getTotalDunklesIncluded', [])
                dunkle_txs = []
                for i, u in enumerate(self.get_uncles()[:4]):
                    start_nonce = self.chain.state.get_nonce(self.address)
                    txdata = casper_ct.encode('includeDunkle', [rlp.encode(u)])
                    dunkle_txs.append(Transaction(start_nonce + i, 0, 650000, self.chain.config['CASPER_ADDR'], 0, txdata).sign(self.key))
                for dtx in dunkle_txs[::-1]:
                    self.chain.add_transaction(dtx, force=True)
                blk = make_block(self.chain, self.key, self.randao, self.indices, self.next_skip_count)
                global global_block_counter
                global_block_counter += 1
                for dtx in dunkle_txs:
                    assert dtx in blk.transactions, (dtx, blk.transactions)
                print 'made block with timestamp %d and %d dunkles' % (blk.timestamp, len(dunkle_txs))
                assert blk.timestamp >= self.next_skip_timestamp
                assert self.chain.add_block(blk)
                self.update_head()
                post_dunkle_count = call_casper(self.chain.state, 'getTotalDunklesIncluded', [])
                assert post_dunkle_count - pre_dunkle_count == len(dunkle_txs)
                self.received_objects[blk.hash] = True
                print 'Validator %d making block %d (%s)' % (self.id, blk.header.number, blk.header.hash[:8].encode('hex'))
                self.network.broadcast(self, blk)
        # Sometimes we received blocks too early or out of order;
        # run an occasional loop that processes these
        if random.random() < 0.02:
            self.chain.process_time_queue()
            self.chain.process_parent_queue()
            self.update_head()

    def update_head(self):
        if self.cached_head == self.chain.head_hash:
            return
        self.cached_head = self.chain.head_hash
        if self.chain.state.block_number % self.epoch_length == 0:
            self.find_my_indices()
        if self.indices:
            self.next_skip_count = 0
            self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
        print 'Head changed: %s, will attempt creating a block at %d' % (self.chain.head_hash.encode('hex'), self.next_skip_timestamp)

    def withdraw(self, gasprice=20 * 10**9):
        h = sha3(b'withdrawwithdrawwithdrawwithdraw')
        v, r, s = ecsign(h, self.key)
        sigdata = encode_int32(v) + encode_int32(r) + encode_int32(s)
        txdata = casper_ct.encode('startWithdrawal', [self.indices[0], self.indices[1], sigdata])
        tx = Transaction(self.chain.state.get_nonce(self.address), gasprice, 650000, self.chain.config['CASPER_ADDR'], 0, txdata).sign(self.key)
        self.chain.add_transaction(tx)
        self.network.broadcast(self, tx)
        print 'Withdrawing!'
