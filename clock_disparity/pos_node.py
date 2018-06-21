import os
from binascii import hexlify
from Crypto.Hash import keccak
import random

def to_hex(s):
    return hexlify(s).decode('utf-8')

memo = {}

def sha3(x):
    if x not in memo:
        memo[x] = keccak.new(digest_bits=256, data=x).digest()
    return memo[x]

def hash_to_int(h):
    o = 0
    for c in h:
        o = (o << 8) + c
    return o

NOTARIES = 20
BASE_TS_DIFF = 1
SKIP_TS_DIFF = 6
SAMPLE = 8
MIN_SAMPLE = lambda x: [6, 4, 3, 2, 2][x] if x < 5 else 1
POWDIFF = 50 * NOTARIES
SHARDS = 12

# Not a full RANDAO; stub for now
class Block():
    def __init__(self, parent, proposer, ts, sigs):
        self.contents = os.urandom(32)
        self.parent_hash = parent.hash if parent else (b'\x11' * 32)
        self.hash = sha3(self.parent_hash + self.contents)
        self.ts = ts
        self.sigs = sigs
        self.number = parent.number + 1 if parent else 0

        if parent:
            i = parent.child_proposers.index(proposer)
            assert self.ts >= parent.ts + BASE_TS_DIFF + i * SKIP_TS_DIFF
            assert len(sigs) >= parent.notary_req
            for sig in sigs:
                assert sig.target_hash == self.parent_hash

        # Calculate child proposers
        v = hash_to_int(sha3(self.contents))
        self.child_proposers = []
        while v > 0:
            self.child_proposers.append(v % NOTARIES)
            v //= NOTARIES

        # Calculate notaries
        if not parent:
            index = 0
        elif proposer in parent.child_proposers:
            index = parent.child_proposers.index(proposer)
        else:
            index = len(parent.child_proposers)
        self.notary_req = MIN_SAMPLE(index)
        v = hash_to_int(sha3(self.contents + b':n'))
        self.notaries = []
        for i in range(SAMPLE):
            self.notaries.append(v % NOTARIES)
            v //= NOTARIES

        # Calculate shard proposers
        v = hash_to_int(sha3(self.contents + b':s'))
        self.shard_proposers = []
        for i in range(SHARDS):
            self.shard_proposers.append(v % NOTARIES)
            v //= NOTARIES
        

class Sig():
    def __init__(self, proposer, target):
        self.proposer = proposer
        self.target_hash = target.hash
        self.hash = os.urandom(32)
        assert self.proposer in target.notaries

genesis = Block(None, 1, 0, [])

class BlockMakingRequest():
    def __init__(self, parent, ts):
        self.parent = parent
        self.ts = ts
        self.hash = os.urandom(32)

class Node():

    def __init__(self, _id, network, sleepy=False, careless=False, ts=0):
        self.blocks = {
            genesis.hash: genesis,
        }
        self.sigs = {}
        self.main_chain = [genesis.hash]
        self.timequeue = []
        self.parentqueue = {}
        self.children = {}
        self.ts = ts
        self.id = _id
        self.network = network
        self.used_parents = {}
        self.processed = {}
        self.sleepy = sleepy
        self.careless = careless
        self.first_round = True

    def broadcast(self, x):
        if self.sleepy and self.ts:
            return
        self.network.broadcast(self, x)
        self.on_receive(x)

    def log(self, words, lvl=3, all=False):
        #if "Tick:" != words[:5] or self.id == 0:
        if (self.id == 0 or all) and lvl >= 2:
            print(self.id, words)

    def on_receive(self, obj, reprocess=False):
        if obj.hash in self.processed and not reprocess:
            return
        self.processed[obj.hash] = obj
        if isinstance(obj, Block):
            return self.on_receive_beacon_block(obj)
        elif isinstance(obj, Sig):
            return self.on_receive_sig(obj)
        elif isinstance(obj, BlockMakingRequest):
            if self.main_chain[-1] == obj.parent:
                mc_ref = self.blocks[obj.parent]
                for i in range(2):
                    if mc_ref.number == 0:
                        break
                    #mc_ref = self.blocks[mc_ref].parent_hash
                x = Block(self.blocks[obj.parent], self.id, self.ts,
                                self.sigs[obj.parent] if obj.parent in self.sigs else [])
                self.log("Broadcasting block %s" % to_hex(x.hash[:4]))
                self.broadcast(x)

    def add_to_timequeue(self, obj):
        i = 0
        while i < len(self.timequeue) and self.timequeue[i].ts < obj.ts: 
            i += 1
        self.timequeue.insert(i, obj)

    def add_to_multiset(self, _set, k, v):
        if k not in _set:
            _set[k] = []
        _set[k].append(v)

    def change_head(self, chain, new_head):
        chain.extend([None] * (new_head.number + 1 - len(chain)))
        i, c = new_head.number, new_head.hash
        while c != chain[i]:
            chain[i] = c
            c = self.blocks[c].parent_hash
            i -= 1
        for i in range(len(chain)):
            assert self.blocks[chain[i]].number == i

    def recalculate_head(self, chain, condition):
        while not condition(self.blocks[chain[-1]]):
            chain.pop()
        descendant_queue = [chain[-1]]
        new_head = chain[-1]
        while len(descendant_queue):
            first = descendant_queue.pop(0)
            if first in self.children:
                for c in self.children[first]:
                    if condition(self.blocks[c]):
                        descendant_queue.append(c)
            if self.blocks[first].number > self.blocks[new_head].number:
                new_head = first
        self.change_head(chain, self.blocks[new_head])
        for i in range(len(chain)):
            assert condition(self.blocks[chain[i]])

    def process_children(self, h):
        if h in self.parentqueue:
            for b in self.parentqueue[h]:
                self.on_receive(b, reprocess=True)
            del self.parentqueue[h]

    def is_descendant(self, a, b):
        a, b = self.blocks[a], self.blocks[b]
        while b.number > a.number:
            b = self.blocks[b.parent_hash]
        return a.hash == b.hash

    def have_ancestry(self, h):
        while h != genesis.hash:
            if h not in self.processed:
                return False
            h = self.processed[h].parent_hash
        return True

    def is_notarized(self, b):
        return len(self.sigs.get(b.hash, [])) >= b.notary_req

    def on_receive_beacon_block(self, block):
        # Parent not yet received
        if block.parent_hash not in self.blocks:
            self.add_to_multiset(self.parentqueue, block.parent_hash, block)
            return
        # Too early
        if block.ts > self.ts:
            self.add_to_timequeue(block)
            return
        # Add the block
        self.log("Processing beacon block %s" % to_hex(block.hash[:4]))
        self.blocks[block.hash] = block
        # Am I a notary, and is the block building on the head? Then broadcast a signature.
        if block.parent_hash == self.main_chain[-1] or self.careless:
            if self.id in block.notaries:
                self.broadcast(Sig(self.id, block))
        # Add child record
        self.add_to_multiset(self.children, block.parent_hash, block.hash)
        # Final steps
        self.process_children(block.hash)
        self.network.broadcast(self, block)

    def on_receive_sig(self, sig):
        if sig.target_hash not in self.blocks:
            self.add_to_multiset(self.parentqueue, sig.target_hash, sig)
            return
        # Add to head? Make a block?
        self.add_to_multiset(self.sigs, sig.target_hash, sig)
        if len(self.sigs[sig.target_hash]) == self.blocks[sig.target_hash].notary_req:
            block = self.blocks[sig.target_hash]
            if block.number > self.blocks[self.main_chain[-1]].number:
                self.change_head(self.main_chain, block)
            if self.id in block.child_proposers:
                my_index = block.child_proposers.index(self.id)
                target_ts = block.ts + BASE_TS_DIFF + my_index * SKIP_TS_DIFF
                self.log("Making block request for %.1f" % target_ts)
                self.add_to_timequeue(BlockMakingRequest(block.hash, target_ts))
        # Rebroadcast
        self.network.broadcast(self, sig)

    def tick(self):
        if self.first_round:
            if self.id in genesis.notaries:
                self.broadcast(Sig(self.id, genesis))
            self.first_round = False
        self.ts += 0.1
        self.log("Tick: %.1f" % self.ts, lvl=1)
        # Process time queue
        while len(self.timequeue) and self.timequeue[0].ts <= self.ts:
            self.on_receive(self.timequeue.pop(0), reprocess=True)
