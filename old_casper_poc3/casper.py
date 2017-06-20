from ethereum.casper_utils import RandaoManager, get_skips_and_block_making_time, \
    generate_validation_code, call_casper, sign_block, check_skips, get_timestamp, \
    get_casper_ct, get_dunkle_candidates, \
    make_withdrawal_signature
from ethereum.utils import sha3, hash32, privtoaddr, ecsign, zpad, encode_int32, \
    big_endian_to_int
from ethereum.transaction_queue import TransactionQueue
from ethereum.block_creation import make_head_candidate
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
        # Create a transaction queue
        self.txqueue = TransactionQueue()
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
        # Is this validator active?
        self.active = False
        # Code that verifies signatures from this validator
        self.validation_code = generate_validation_code(privtoaddr(key))
        # Validation code hash
        self.vchash = sha3(self.validation_code)
        # Parents that this validator has already built a block on
        self.used_parents = {}
        # This validator's clock offset (for testing purposes)
        self.time_offset = random.randrange(time_offset) - (time_offset // 2)
        # Determine the epoch length
        self.epoch_length = self.call_casper('getEpochLength')
        # My minimum gas price
        self.mingasprice = 20 * 10**9
        # Give this validator a unique ID
        self.id = len(ids)
        ids.append(self.id)
        self.update_activity_status()
        self.cached_head = self.chain.head_hash

    def call_casper(self, fun, args=[]):
        return call_casper(self.chain.state, fun, args)

    def update_activity_status(self):
        start_epoch = self.call_casper('getStartEpoch', [self.vchash])
        now_epoch = self.call_casper('getEpoch')
        end_epoch = self.call_casper('getEndEpoch', [self.vchash])
        if start_epoch <= now_epoch < end_epoch:
            self.active = True
            self.next_skip_count = 0
            self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
            print 'In current validator set'
        else:
            self.active = False

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
            print 'Receiving transaction', obj
            if obj.gasprice >= self.mingasprice:
                self.txqueue.add_transaction(obj)
                print 'Added transaction, txqueue size %d' % len(self.txqueue.txs)
                self.network.broadcast(self, obj)
            else:
                print 'Gasprice too low'
        self.received_objects[obj.hash] = True
        for x in self.chain.get_chain():
            assert x.hash in self.received_objects

    def tick(self):
        # Try to create a block
        # Conditions:
        # (i) you are an active validator,
        # (ii) you have not yet made a block with this parent
        if self.active and self.chain.head_hash not in self.used_parents:
            t = self.get_timestamp()
            # Is it early enough to create the block?
            if t >= self.next_skip_timestamp and (not self.chain.head or t > self.chain.head.header.timestamp):
                # Wrong validator; in this case, just wait for the next skip count
                if not check_skips(self.chain, self.vchash, self.next_skip_count):
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
                # Make the block
                s1 = self.chain.state.trie.root_hash
                pre_dunkle_count = self.call_casper('getTotalDunklesIncluded')
                dunkle_txs = get_dunkle_candidates(self.chain, self.chain.state)
                blk = make_head_candidate(self.chain, self.txqueue)
                randao = self.randao.get_parent(self.call_casper('getRandao', [self.vchash]))
                blk = sign_block(blk, self.key, randao, self.vchash, self.next_skip_count)
                # Make sure it's valid
                global global_block_counter
                global_block_counter += 1
                for dtx in dunkle_txs:
                    assert dtx in blk.transactions, (dtx, blk.transactions)
                print 'made block with timestamp %d and %d dunkles' % (blk.timestamp, len(dunkle_txs))
                s2 = self.chain.state.trie.root_hash
                assert s1 == s2
                assert blk.timestamp >= self.next_skip_timestamp
                assert self.chain.add_block(blk)
                self.update_head()
                post_dunkle_count = self.call_casper('getTotalDunklesIncluded')
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
            self.update_activity_status()
        if self.active:
            self.next_skip_count = 0
            self.next_skip_timestamp = get_timestamp(self.chain, self.next_skip_count)
        print 'Head changed: %s, will attempt creating a block at %d' % (self.chain.head_hash.encode('hex'), self.next_skip_timestamp)

    def withdraw(self, gasprice=20 * 10**9):
        sigdata = make_withdrawal_signature(self.key)
        txdata = casper_ct.encode('startWithdrawal', [self.vchash, sigdata])
        tx = Transaction(self.chain.state.get_nonce(self.address), gasprice, 650000, self.chain.config['CASPER_ADDR'], 0, txdata).sign(self.key)
        self.txqueue.add_transaction(tx, force=True)
        self.network.broadcast(self, tx)
        print 'Withdrawing!'

    def deposit(self, gasprice=20 * 10**9):
        assert value * 10**18 >= self.chain.state.get_balance(self.address) + gasprice * 1000000
        tx = Transaction(self.chain.state.get_nonce(self.address) * 10**18, gasprice, 1000000,
                         casper_config['CASPER_ADDR'], value * 10**18,
                         ct.encode('deposit', [self.validation_code, self.randao.get(9999)]))
