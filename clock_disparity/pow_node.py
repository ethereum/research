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

NOTARIES = 40
BASE_TS_DIFF = 1
SKIP_TS_DIFF = 6
SAMPLE = 8
MIN_SAMPLE = 7
POWDIFF = 50 * NOTARIES
SHARDS = 12

def checkpow(work, nonce):
    # Discrete log PoW, lolz
    # Quadratic nonresidues only
    return pow(work, nonce, 65537) * POWDIFF < 65537 * 2 and pow(nonce, 32768, 65537) == 65536

class Block():
    def __init__(self, parent, pownonce, ts):
        self.parent_hash = parent.hash if parent else (b'\x00' * 32)
        assert isinstance(self.parent_hash, bytes)
        self.hash = sha3(self.parent_hash + str(pownonce).encode('utf-8'))
        self.ts = ts
        if parent:
            assert checkpow(parent.pownonce, pownonce)
            assert self.ts >= parent.ts
        self.pownonce = pownonce
        self.number = 0 if parent is None else parent.number + 1
            

genesis = Block(None, 59049, 0)

class Node():

    def __init__(self, _id, network, sleepy=False, careless=False, ts=0):
        self.blocks = {
            genesis.hash: genesis
        }
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
            return self.on_receive_main_block(obj)

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
            assert self.blocks[chain[i]].ts <= self.ts

    def process_children(self, h):
        if h in self.parentqueue:
            for b in self.parentqueue[h]:
                self.on_receive(b, reprocess=True)
            del self.parentqueue[h]

    def have_ancestry(self, h):
        while h != genesis.hash:
            if h not in self.processed:
                return False
            h = self.processed[h].parent_hash
        return True

    def is_notarized(self, b):
        return b.hash in self.children

    def on_receive_main_block(self, block):
        # Parent not yet received
        if block.parent_hash not in self.blocks:
            self.add_to_multiset(self.parentqueue, block.parent_hash, block)
            return None
        if block.ts > self.ts:
            self.add_to_timequeue(block)
            return None
        self.log("Processing main chain block %s" % to_hex(block.hash[:4]))
        self.blocks[block.hash] = block
        # Reorg the main chain if new head
        if block.number > self.blocks[self.main_chain[-1]].number:
            reorging = (block.parent_hash != self.main_chain[-1])
            self.change_head(self.main_chain, block)
        # Add child record
        self.add_to_multiset(self.children, block.parent_hash, block.hash)
        # Final steps
        self.process_children(block.hash)
        self.network.broadcast(self, block)

    def is_descendant(self, a, b):
        a, b = self.blocks[a], self.blocks[b]
        while b.number > a.number:
            b = self.blocks[b.parent_hash]
        return a.hash == b.hash

    def tick(self):
        self.ts += 0.1
        self.log("Tick: %.1f" % self.ts, lvl=1)
        # Process time queue
        while len(self.timequeue) and self.timequeue[0].ts <= self.ts:
            self.on_receive(self.timequeue.pop(0), reprocess=True)
        # Attempt to mine a main chain block
        pownonce = random.randrange(65537)
        mchead = self.blocks[self.main_chain[-1]]
        if checkpow(mchead.pownonce, pownonce):
            assert self.ts >= mchead.ts
            self.broadcast(Block(mchead, pownonce, self.ts))
