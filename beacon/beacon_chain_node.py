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
SAMPLE = 9
MIN_SAMPLE = 5
POWDIFF = 30 * NOTARIES

def checkpow(work, nonce):
    # Discrete log PoW, lolz
    # Quadratic nonresidues only
    return pow(work, nonce, 65537) * POWDIFF < 65537 * 2 and pow(nonce, 32768, 65537) == 65536

class MainChainBlock():
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
            

# Not a full RANDAO; stub for now
class BeaconBlock():
    def __init__(self, parent, proposer, ts, sigs, main_chain_ref):
        self.contents = os.urandom(32)
        self.parent_hash = parent.hash if parent else (b'\x11' * 32)
        self.hash = sha3(self.parent_hash + self.contents)
        self.ts = ts
        self.sigs = sigs
        self.number = parent.number + 1 if parent else 0
        self.main_chain_ref = main_chain_ref if main_chain_ref else parent.main_chain_ref

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
        first = parent and proposer == parent.child_proposers[0]
        self.notary_req = 0 if first else MIN_SAMPLE
        v = hash_to_int(sha3(self.contents + b':n'))
        self.notaries = []
        for i in range(SAMPLE if first else SAMPLE):
            self.notaries.append(v % NOTARIES)
            v //= NOTARIES


class Sig():
    def __init__(self, proposer, target):
        self.proposer = proposer
        self.target_hash = target.hash
        self.hash = os.urandom(32)
        assert self.proposer in target.notaries

main_genesis = MainChainBlock(None, 59049, 0)
beacon_genesis = BeaconBlock(None, 1, 0, [], main_genesis.hash)

class BlockMakingRequest():
    def __init__(self, parent, ts):
        self.parent = parent
        self.ts = ts
        self.hash = os.urandom(32)

class Node():

    def __init__(self, _id, network, sleepy=False, careless=False):
        self.blocks = {beacon_genesis.hash: beacon_genesis, main_genesis.hash: main_genesis}
        self.sigs = {}
        self.beacon_head = beacon_genesis.hash
        self.main_chain = [main_genesis.hash]
        self.timequeue = []
        self.parentqueue = {}
        self.children = {}
        self.ts = 0
        self.id = _id
        self.network = network
        self.used_parents = {}
        self.processed = {}
        self.sleepy = sleepy
        self.careless = careless

    def broadcast(self, x):
        if self.sleepy and self.ts:
            return
        #self.log("Broadcasting %s %s" % ("block" if isinstance(x, BeaconBlock) else "sig", to_hex(x.hash[:4])))
        self.network.broadcast(self, x)
        self.on_receive(x)

    def log(self, words, lvl=3, all=False):
        #if "Tick:" != words[:5] or self.id == 0:
        if (self.id == 0 or all) and lvl >= 2:
            print(self.id, words)

    def on_receive(self, obj, reprocess=False):
        if obj.hash in self.processed and not reprocess:
            return
        self.processed[obj.hash] = True
        #self.log("Processing %s %s" % ("block" if isinstance(obj, BeaconBlock) else "sig", to_hex(obj.hash[:4])))
        if isinstance(obj, BeaconBlock):
            return self.on_receive_block(obj)
        elif isinstance(obj, MainChainBlock):
            return self.on_receive_main_block(obj)
        elif isinstance(obj, Sig):
            return self.on_receive_sig(obj)
        elif isinstance(obj, BlockMakingRequest):
            if self.beacon_head == obj.parent:
                mc_ref = self.blocks[obj.parent]
                for i in range(2):
                    if mc_ref.number == 0:
                        break
                    #mc_ref = self.blocks[mc_ref].parent_hash
                x = BeaconBlock(self.blocks[obj.parent], self.id, self.ts,
                                self.sigs[obj.parent] if obj.parent in self.sigs else [],
                                self.main_chain[-1])
                self.log("Broadcasting block %s" % to_hex(x.hash[:4]))
                self.broadcast(x)

    def add_to_timequeue(self, obj):
        i = 0
        while i < len(self.timequeue) and self.timequeue[i].ts < obj.ts: 
            i += 1
        self.timequeue.insert(i, obj)

    def on_receive_main_block(self, block):
        # Parent not yet received
        if block.parent_hash not in self.blocks:
            if block.parent_hash not in self.parentqueue:
                self.parentqueue[block.parent_hash] = []
            self.parentqueue[block.parent_hash].append(block)
            return None
        self.log("Processing main chain block %s" % to_hex(block.hash[:4]))
        self.blocks[block.hash] = block
        # Reorg the main chain if new head
        if block.number > self.blocks[self.main_chain[-1]].number:
            assert block.number == len(self.main_chain), (block.number, self.blocks[self.main_chain[-1]].number)
            reorging = (block.parent_hash != self.main_chain[-1])
            if reorging:
                self.log("Reorging main chain", all=True)
            i, c = block.number - 1, block.parent_hash
            while c != self.main_chain[i]:
                self.main_chain[i] = c
                c = self.blocks[c].parent_hash
                i -= 1
            self.main_chain.append(block.hash)
            for i in range(len(self.main_chain)):
                assert self.blocks[self.main_chain[i]].number == i
            # Reorg the beacon
            if reorging:
                pre_beacon = self.beacon_head
                while self.blocks[self.beacon_head].main_chain_ref not in self.main_chain:
                    self.beacon_head = self.blocks[self.beacon_head].parent_hash
                descendant_queue = [self.beacon_head]
                while len(descendant_queue):
                    first = descendant_queue.pop(0)
                    if first in self.children:
                        for c in self.children[first]:
                            if isinstance(self.blocks[c], BeaconBlock) and self.blocks[c].main_chain_ref in self.main_chain:
                                descendant_queue.append(c)
                    if self.blocks[first].number > self.blocks[self.beacon_head].number:
                        self.beacon_head = first
                if self.beacon_head != pre_beacon:
                    self.log("Reorged beacon due to main chain reorg", all=True)
        # Add child record
        if block.parent_hash not in self.children:
            self.children[block.parent_hash] = []
        self.children[block.parent_hash].append(block.hash)
        # Check for children
        if block.hash in self.parentqueue:
            for b in self.parentqueue[block.hash]:
                self.on_receive(b, reprocess=True)
            del self.parentqueue[block.hash]
        self.network.broadcast(self, block)

    def is_descendant(self, a, b):
        a, b = self.blocks[a], self.blocks[b]
        while b.number > a.number:
            b = self.blocks[b.parent_hash]
        return a.hash == b.hash

    def on_receive_block(self, block):
        # Parent not yet received
        if block.parent_hash not in self.blocks:
            if block.parent_hash not in self.parentqueue:
                self.parentqueue[block.parent_hash] = []
            self.parentqueue[block.parent_hash].append(block)
            return None
        # Main chain parent not yet received
        if block.main_chain_ref not in self.blocks:
            if block.main_chain_ref not in self.parentqueue:
                self.parentqueue[block.main_chain_ref] = []
            self.parentqueue[block.main_chain_ref].append(block)
            return None
        # Too early
        if block.ts > self.ts:
            self.add_to_timequeue(block)
            return
        assert block.parent_hash in self.blocks
        assert block.main_chain_ref in self.blocks
        assert self.blocks[block.parent_hash].main_chain_ref in self.blocks
        # Check consistency of cross-link reference
        assert self.is_descendant(self.blocks[block.parent_hash].main_chain_ref, block.main_chain_ref)
        # Add the block
        self.log("Processing beacon block %s" % to_hex(block.hash[:4]))
        self.blocks[block.hash] = block
        # Am I a notary, and is the block building on the head?
        # careless = I notarize even stuff not on the head
        if block.parent_hash == self.beacon_head or self.careless:
            if self.id in block.notaries:
                # Then broadcast a signature
                self.broadcast(Sig(self.id, block))
        # Check for sigs, add to head?, make a block?
        if (block.hash in self.sigs and len(self.sigs[block.hash]) >= block.notary_req) or block.notary_req == 0:
            if block.number > self.blocks[self.beacon_head].number and block.main_chain_ref in self.main_chain:
                self.log("Changed head: %s" % block.number)
                self.beacon_head = block.hash
            if self.id in self.blocks[block.hash].child_proposers:
                my_index = self.blocks[block.hash].child_proposers.index(self.id)
                target_ts = block.ts + BASE_TS_DIFF + my_index * SKIP_TS_DIFF
                self.log("Making block request for %.1f" % target_ts)
                self.add_to_timequeue(BlockMakingRequest(block.hash, target_ts))
        # Add child record
        if block.parent_hash not in self.children:
            self.children[block.parent_hash] = []
        self.children[block.parent_hash].append(block.hash)
        # Check for children
        if block.hash in self.parentqueue:
            for b in self.parentqueue[block.hash]:
                self.on_receive(b, reprocess=True)
            del self.parentqueue[block.hash]
        # Rebroadcast
        self.network.broadcast(self, block)

    def on_receive_sig(self, sig):
        self.log("Processing sig for %s" % to_hex(sig.target_hash[:4]), lvl=1)
        if sig.target_hash not in self.sigs:
            self.sigs[sig.target_hash] = []
        self.sigs[sig.target_hash].append(sig)
        # Add to head? Make a block?
        if sig.target_hash in self.blocks and len(self.sigs[sig.target_hash]) == self.blocks[sig.target_hash].notary_req:
            block = self.blocks[sig.target_hash]
            if block.number > self.blocks[self.beacon_head].number and block.main_chain_ref in self.main_chain:
                self.log("Changed head: %s" % block.number)
                self.beacon_head = block.hash
            if self.id in block.child_proposers:
                my_index = block.child_proposers.index(self.id)
                target_ts = block.ts + BASE_TS_DIFF + my_index * SKIP_TS_DIFF
                self.log("Making block request for %.1f" % target_ts)
                self.add_to_timequeue(BlockMakingRequest(block.hash, target_ts))
        # Rebroadcast
        self.network.broadcast(self, sig)

    def tick(self):
        if self.ts == 0:
            if self.id in beacon_genesis.notaries:
                self.broadcast(Sig(self.id, beacon_genesis))
        self.ts += 0.1
        self.log("Tick: %.1f" % self.ts, lvl=1)
        while len(self.timequeue) and self.timequeue[0].ts <= self.ts:
            self.on_receive(self.timequeue.pop(0))
        pownonce = random.randrange(65537)
        mchead = self.blocks[self.main_chain[-1]]
        if checkpow(mchead.pownonce, pownonce):
            self.broadcast(MainChainBlock(mchead, pownonce, self.ts))
