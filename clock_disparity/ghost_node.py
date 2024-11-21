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

NOTARIES = 100
SLOT_SIZE = 6
EPOCH_LENGTH = 25

# Not a full RANDAO; stub for now
class Block():
    def __init__(self, parent, slot, proposer):
        self.contents = os.urandom(32)
        self.parent_hash = parent.hash if parent else (b'\x00' * 32)
        self.hash = sha3(self.parent_hash + self.contents)
        self.height = parent.height + 1 if parent else 0
        assert slot % NOTARIES == proposer
        self.proposer = proposer
        self.slot = slot

    def min_timestamp(self):
        return SLOT_SIZE * self.slot

class Sig():
    def __init__(self, proposer, targets, slot, ts):
        self.proposer = proposer
        self.targets = targets
        self.slot = slot
        self.ts = ts
        self.hash = os.urandom(32)

genesis = Block(None, 0, 0)

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
        self.scores = {}
        self.scores_at_height = {}
        self.justified = {}
        self.finalized = {}
        self.ts = ts
        self.id = _id
        self.network = network
        self.used_parents = {}
        self.processed = {}
        self.sleepy = sleepy
        self.careless = careless
        self.first_round = True
        self.last_made_block = 0
        self.last_made_sig = 0

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

    def add_to_timequeue(self, obj):
        i = 0
        while i < len(self.timequeue) and self.timequeue[i].min_timestamp() < obj.min_timestamp(): 
            i += 1
        self.timequeue.insert(i, obj)

    def add_to_multiset(self, _set, k, v):
        if k not in _set:
            _set[k] = []
        _set[k].append(v)

    def change_head(self, chain, new_head):
        chain.extend([None] * (new_head.height + 1 - len(chain)))
        i, c = new_head.height, new_head.hash
        while c != chain[i]:
            chain[i] = c
            c = self.blocks[c].parent_hash
            i -= 1
        for i in range(len(chain)):
            assert self.blocks[chain[i]].height == i

    def recalculate_head(self):
        while 1:
            descendant_queue = [self.main_chain[-1]]
            new_head = None
            max_count = 0
            while len(descendant_queue):
                first = descendant_queue.pop(0)
                if first in self.children:
                    for c in self.children[first]:
                        descendant_queue.append(c)
                if self.scores.get(first, 0) > max_count and first != self.main_chain[-1]:
                    new_head = first
                    max_count = self.scores.get(first, 0)
            if new_head:
                self.change_head(self.main_chain, self.blocks[new_head])
            else:
                return

    def process_children(self, h):
        if h in self.parentqueue:
            for b in self.parentqueue[h]:
                self.on_receive(b, reprocess=True)
            del self.parentqueue[h]

    def get_common_ancestor(self, a, b):
        a, b = self.blocks[a], self.blocks[b]
        while b.height > a.height:
            b = self.blocks[b.parent_hash]
        while a.height > b.height:
            a = self.blocks[a.parent_hash]
        while a.hash != b.hash:
            a = self.blocks[a.parent_hash]
            b = self.blocks[b.parent_hash]
        return a
        
    def is_descendant(self, a, b):
        a, b = self.blocks[a], self.blocks[b]
        while b.height > a.height:
            b = self.blocks[b.parent_hash]
        return a.hash == b.hash

    def have_ancestry(self, h):
        while h != genesis.hash:
            if h not in self.processed:
                return False
            h = self.processed[h].parent_hash
        return True

    def on_receive_beacon_block(self, block):
        # Parent not yet received
        if block.parent_hash not in self.blocks:
            self.add_to_multiset(self.parentqueue, block.parent_hash, block)
            return
        # Too early
        if block.min_timestamp() > self.ts:
            self.add_to_timequeue(block)
            return
        # Add the block
        self.log("Processing beacon block %s" % to_hex(block.hash[:4]))
        self.blocks[block.hash] = block
        # Is the block building on the head? Then add it to the head!
        if block.parent_hash == self.main_chain[-1] or self.careless:
            self.main_chain.append(block.hash)
        # Add child record
        self.add_to_multiset(self.children, block.parent_hash, block.hash)
        # Final steps
        self.process_children(block.hash)
        self.network.broadcast(self, block)

    def on_receive_sig(self, sig):
        if sig.targets[0] not in self.blocks:
            self.add_to_multiset(self.parentqueue, sig.targets[0], sig)
            return
        # Get common ancestor
        anc = self.get_common_ancestor(self.main_chain[-1], sig.targets[0]) 
        max_score = max([0] + [self.scores.get(self.main_chain[i], 0) for i in range(anc.height + 1, len(self.main_chain))])
        # Process scoring
        max_newchain_score = 0
        for i, c in list(enumerate(sig.targets))[::-1]:
            slot = sig.slot - 1 - i
            slot_key = slot.to_bytes(4, 'big')
            assert self.blocks[c].slot <= slot

            # If a parent and child block have non-consecutive slots, then the parent
            # block is also considered to be the canonical block at all of the intermediate
            # slot numbers. We store the scores for the block at each height separately
            self.scores_at_height[slot_key + c] = self.scores_at_height.get(slot_key + c, 0) + 1

            # For fork choice rule purposes, the score of a block is the highest score
            # that it has at any height
            self.scores[c] = max(self.scores.get(c, 0), self.scores_at_height[slot_key + c])

            # If 2/3 of notaries vote for a block, it is justified
            if self.scores_at_height[slot_key + c] == NOTARIES * 2 // 3:
                self.justified[c] = True
                c2 = c
                self.log("Justified: %d %s" % (slot, hexlify(c).decode('utf-8')[:8]))

                # If EPOCH_LENGTH+1 blocks are justified in a row, the oldest is
                # considered finalized

                finalize = True
                for slot2 in range(slot - 1, max(slot - EPOCH_LENGTH * 2, 0) - 1, -1):
                    if slot2 < self.blocks[c2].slot:
                        c2 = self.blocks[c2].parent_hash
                    if self.scores_at_height.get(slot2.to_bytes(4, 'big') + c2, 0) < (NOTARIES * 2 // 3):
                        finalize = False
                        # self.log("Not quite finalized: stopped at %d needed %d" % (slot2, max(slot - EPOCH_LENGTH, 0)))
                        break
                    if slot2 < slot - EPOCH_LENGTH - 1 and finalize and c2 not in self.finalized:
                        self.log("Finalized: %d %s" % (self.blocks[c2].slot, hexlify(c).decode('utf-8')[:8]))
                        self.finalized[c2] = True

            # Find the maximum score of a block on the chain that this sig is weighing on
            if self.blocks[c].slot > anc.slot:
                max_newchain_score = max(max_newchain_score, self.scores[c])

        # If it's higher, switch over the canonical chain
        if max_newchain_score > max_score:
            self.main_chain = self.main_chain[:anc.height+1]
            self.recalculate_head()

        self.sigs[sig.hash] = sig

        # Rebroadcast
        self.network.broadcast(self, sig)

    # Get the portion of the main chain that is within the last EPOCH_LENGTH
    # slots, once again duplicating the parent in cases where the parent and
    # child's slots are not consecutive
    def get_sig_targets(self, start_slot):
        o = []
        i = len(self.main_chain) - 1
        for slot in range(start_slot - 1, max(start_slot - EPOCH_LENGTH, 0) - 1, -1):
            if slot < self.blocks[self.main_chain[i]].slot:
                i -= 1
            o.append(self.main_chain[i])
        for i, x in enumerate(o):
            assert self.blocks[x].slot <= start_slot - 1 - i
        assert len(o) == min(EPOCH_LENGTH, start_slot)
        return o

    def tick(self):
        self.ts += 0.1
        self.log("Tick: %.1f" % self.ts, lvl=1)
        # Make a block?
        slot = int(self.ts // SLOT_SIZE)
        if slot > self.last_made_block and (slot % NOTARIES) == self.id:
            self.broadcast(Block(self.blocks[self.main_chain[-1]], slot, self.id))
            self.last_made_block = slot
        # Make a sig?
        if slot > self.last_made_sig and (slot % EPOCH_LENGTH) == self.id % EPOCH_LENGTH:
            sig_from = len(self.main_chain) - 1
            while sig_from > 0 and self.blocks[self.main_chain[sig_from]].slot >= slot - EPOCH_LENGTH:
                sig_from -= 1
            sig = Sig(self.id, self.get_sig_targets(slot), slot, self.ts)
            # self.log('Sig:', self.id, sig.slot, ' '.join([hexlify(t).decode('utf-8')[:4] for t in sig.targets]))
            self.broadcast(sig)
            self.last_made_sig = slot
        # Process time queue
        while len(self.timequeue) and self.timequeue[0].min_timestamp() <= self.ts:
            self.on_receive(self.timequeue.pop(0), reprocess=True)
