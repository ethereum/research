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

def get_most_common_entry(lst):
    counts = {}
    for l in lst:
        counts[l] = counts.get(l, 0) + 1
    maxcount, maxkey = max(zip(counts.values(), counts.keys()))
    return maxkey, maxcount

NOTARIES = 50
SLOT_SIZE = 3
EPOCH_LENGTH = 25

# Not a full RANDAO; stub for now
class Block():
    def __init__(self, parent, slot, proposer):
        self.contents = os.urandom(32)
        if parent:
            self.ancestor_hashes = [None] * 16
            self.ancestor_slots = [0] * 16
            for i in range(16):
                if (parent.slot // 2**i) > (parent.ancestor_slots[i] // 2**i):
                    self.ancestor_hashes[i] = parent.hash
                    self.ancestor_slots[i] = parent.slot
                else:
                    self.ancestor_hashes[i] = parent.ancestor_hashes[i]
                    self.ancestor_slots[i] = parent.ancestor_slots[i]
        else:
            self.ancestor_hashes = [b'\x00' * 32 for i in range(16)]
            self.ancestor_slots = [-1 for i in range(16)]
        self.hash = sha3(self.parent_hash + self.contents)
        self.height = parent.height + 1 if parent else 0
        assert slot % NOTARIES == proposer
        self.proposer = proposer
        self.slot = slot

    @property
    def parent_hash(self):
        return self.ancestor_hashes[0]

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
        self.lmd_head = genesis.hash
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
        self.most_recent_votes = {}
        self.observed_ts_deltas = [0] * NOTARIES

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

    def get_ancestor_at_slot(self, block, slot):
        if not isinstance(block, Block):
            block = self.blocks[block]
        if block.slot <= slot:
            return block
        d = 15
        while (block.slot - slot) < 2**d:
            d -= 1
        anc = self.blocks[block.ancestor_hashes[d]]
        return self.get_ancestor_at_slot(anc, slot)

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
        if not isinstance(a, Block):
            a = self.blocks[a]
        if not isinstance(b, Block):
            b = self.blocks[b]
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
        # head = self.compute_lmd_head()
        # if head.hash == block.hash:
        self.observed_ts_deltas[block.proposer] = block.min_timestamp() - self.ts
        # Final steps
        self.process_children(block.hash)
        self.network.broadcast(self, block)

    def compute_lmd_head(self):
        voters = list(range(NOTARIES))
        binary_search_anchor = 0
        skip = 1
        last_nonnull_maxkey = genesis.hash
        print("Computing LMD head")
        print('top_slots of votes', [self.blocks[v[0]].slot for v in self.most_recent_votes.values()])
        while 1:
            slot_at = binary_search_anchor + skip
            votes_at_slot = []
            voters_for_hash = {}
            for voter in voters:
                if voter in self.most_recent_votes and self.most_recent_votes[voter][1] >= slot_at:
                    votes_at_slot.append(self.get_ancestor_at_slot(self.most_recent_votes[voter][0], slot_at).hash)
            maxkey, maxcount = get_most_common_entry(votes_at_slot) if votes_at_slot else (None, 0)
            if maxkey:
                last_nonnull_maxkey = maxkey
            assert votes_at_slot.count(maxkey) == maxcount
            print(slot_at, maxkey, maxcount)
            if maxcount > len(voters) // 2:
                binary_search_anchor += skip
                skip *= 2
            else:
                if skip == 1:
                    remaining_voters = [
                        v for v in voters if \
                        v in self.most_recent_votes and \
                        self.most_recent_votes[v][1] >= slot_at and \
                        self.get_ancestor_at_slot(self.most_recent_votes[v][0], slot_at).hash == maxkey
                    ]
                    assert maxcount == len(remaining_voters)
                    voters = remaining_voters
                    binary_search_anchor += skip
                    print("%d remaining_voters" % len(remaining_voters))
                    print('top_slots', [self.blocks[self.most_recent_votes[v][0]].slot for v in voters])
                else:
                    skip //= 2
            if len(voters) == 0:
                o = self.blocks[last_nonnull_maxkey]
                while o.hash in self.children:
                    o = self.blocks[self.children[o.hash][0]]
                return o

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

        # Adjust most recent votes array
        existing_vote_slot = self.most_recent_votes.get(sig.proposer, (None, -1))[1]
        if sig.slot > existing_vote_slot:
            self.most_recent_votes[sig.proposer] = (sig.targets[0], sig.slot)

        self.sigs[sig.hash] = sig

        # Rebroadcast
        self.network.broadcast(self, sig)

    # Get an EPOCH_LENGTH-chain starting from a given block and slot
    # slots, once again duplicating the parent in cases where the parent and
    # child's slots are not consecutive
    def get_sig_targets(self, head, slot):
        return [self.get_ancestor_at_slot(head, s).hash for s in range(slot - 1, max(slot - EPOCH_LENGTH, 0) - 1, -1)]

    def get_adjusted_timestamp(self):
        pull_threshold = 0.50
        add_zeroes = int(NOTARIES * (pull_threshold * 2 -1))
        index = int(NOTARIES * pull_threshold)
        return self.ts + sorted(self.observed_ts_deltas + [0] * add_zeroes)[index]

    def tick(self):
        self.ts += 0.1
        self.log("Tick: %.1f" % self.ts, lvl=1)
        # Make a block?
        ts = self.get_adjusted_timestamp()
        slot = int(ts // SLOT_SIZE)
        if slot > self.last_made_block and (slot % NOTARIES) == self.id:
            head = self.compute_lmd_head()
            b = Block(head, slot, self.id)
            self.broadcast(b)
            self.last_made_block = slot
        # Make a sig?
        if slot > self.last_made_sig and (slot % EPOCH_LENGTH) == self.id % EPOCH_LENGTH:
            head = self.compute_lmd_head()
            sig = Sig(self.id, self.get_sig_targets(head, slot), slot, ts)
            # self.log('Sig:', self.id, sig.slot, ' '.join([hexlify(t).decode('utf-8')[:4] for t in sig.targets]))
            self.broadcast(sig)
            self.last_made_sig = slot
        # Process time queue
        while len(self.timequeue) and self.timequeue[0].min_timestamp() <= ts:
            self.on_receive(self.timequeue.pop(0), reprocess=True)
