import copy, random, hashlib
# GhostTable: { block number: { block: validators } }
# The ghost table represents the entire current "view" of a user, and
# every block produced contains the producer's ghost table at the time.

# Signature slashing rules (not implemented)
# 1. Sign two blocks at the same height
# 2. Sign an invalid block
# 3. Sign a block which fully confirms A at height H, and sign B at height H

h = [3**50]
ids = [0]

# Number of validators
NUM_VALIDATORS = 50
# Block time in ticks (eg. 1 tick = 0.1 seconds)
BLKTIME = 30
# Disparity in the blocks of nodes
CLOCK_DISPARITY = 10
# An exponential distribution for latency offset by a minimum
LATENCY_MIN = 4
LATENCY_BASE = 2
LATENCY_PROB = 0.25


def assign_hash():
    h[0] = (h[0] * 3) % 2**48
    return h[0]


def assign_id():
    ids[0] += 1
    return ids[0] - 1


def latency_distribution_sample():
    v = LATENCY_BASE
    while random.random() > LATENCY_PROB:
        v *= 2
    return v + LATENCY_MIN


def clock_offset_distribution_sample():
    return random.randrange(-CLOCK_DISPARITY, CLOCK_DISPARITY)


# A signature represents the entire "view" of a signer,
# where the view is the set of blocks that the signer
# considered most likely to be valid at the time that
# they were produced
class Signature():

    def __init__(self, signer, view):
        self.signer = signer
        self.view = copy.deepcopy(view)


# A ghost table represents the view that a user had of the signatures
# available at the time that the block was produced.
class GhostTable():

    def __init__(self):
        self.confirmed = []
        self.unconfirmed = []

    def process_signature(self, sig):
        # Process every block height in the signature
        for i in range(len(self.confirmed), len(sig.view)):
            # A ghost table entry at a height is a mapping of
            # block hash -> signers
            if i >= len(self.unconfirmed):
                self.unconfirmed.append({})
            cur_entry = self.unconfirmed[i]
            # If the block hash is not yet in the ghost table, add it, and
            # initialize it with an empty signer set
            if sig.view[i] not in cur_entry:
                cur_entry[sig.view[i]] = {}
            # Add the signer
            cur_entry[sig.view[i]][sig.signer] = True
            # If it has 67% signatures, finalize
            if len(cur_entry[sig.view[i]]) > NUM_VALIDATORS * 2 / 3:
                prevgt = block_map[sig.view[i]].gt
                print 'confirmed', block_map[sig.view[i]].height, sig.view[i], prevgt.hash()
                # Update blocks between the previous confirmation and the
                # current confirmation based on the newly confirmed block's
                # ghost table
                for j in range(len(self.confirmed), i):
                    # At each intermediate height, add the block for which we
                    # havethe most signatures
                    maxkey, maxval = 0, 0
                    for k in prevgt.unconfirmed[j]:
                        if len(prevgt.unconfirmed[j][k]) > maxval:
                            maxkey, maxval = k, len(prevgt.unconfirmed[j][k])
                    self.confirmed.append(maxkey)
                    print j, {k: len(prevgt.unconfirmed[j][k]) for k in prevgt.unconfirmed[j]}
                # Then add the new block that got 67% signatures
                print i, sig.view[i]
                self.confirmed.append(sig.view[i])

    def hash(self):
        print hashlib.sha256(repr(self.unconfirmed) +
                             repr(self.confirmed)).hexdigest()[:15]

    # Create a new ghost table that appends to an existing ghost table, adding
    # some set of signatures
    def append(self, sigs):
        x = GhostTable()
        x.confirmed = copy.deepcopy(self.confirmed)
        x.unconfirmed = copy.deepcopy(self.unconfirmed)
        for sig in sigs:
            x.process_signature(sig)
        return x


class Block():

    def __init__(self, h, gt, maker):
        self.gt = gt
        self.height = h
        self.maker = maker
        self.hash = assign_hash()


class Validator():

    def __init__(self):
        self.gt = GhostTable()
        self.view = []
        self.id = assign_id()
        self.new_sigs = []
        self.clock_offset = clock_offset_distribution_sample()
        self.last_block_produced = -99999
        self.last_unseen = 0

    # Is this block compatible with our view?
    def is_compatible_with_view(self, block):
        return block.height >= len(self.view) or \
            self.view[block.height] is None

    # Add a block to this validator's view of probably valid
    # blocks
    def add_to_view(self, block):
        while len(self.view) <= block.height:
            self.view.append(None)
        self.view[block.height] = block.hash
        while self.last_unseen < len(self.view) and \
                self.view[self.last_unseen] is not None:
            self.last_unseen += 1

    # Make a block
    def produce_block(self):
        self.gt = self.gt.append(self.new_sigs)
        newblk = Block(self.last_unseen, self.gt, self.id)
        print 'newblk', newblk.height
        self.add_to_view(newblk)
        publish(newblk)
        newsig = Signature(self.id, self.view[:self.last_unseen])
        self.new_sigs = [newsig]
        publish(newsig)

    # Callback function upon receiving a block
    def on_receive(self, obj):
        if isinstance(obj, Block):
            desired_maker = (self.time() // BLKTIME) % NUM_VALIDATORS
            if 0 <= (desired_maker - obj.maker) % 100 <= 0:
                if self.is_compatible_with_view(obj):
                    self.add_to_view(obj)
                    publish(Signature(self.id, self.view[:self.last_unseen]))
        if isinstance(obj, Signature):
            self.new_sigs.append(obj)

    # Do everything that you need to do in this particular round
    def tick(self):
        if (self.time() // BLKTIME) % NUM_VALIDATORS == self.id:
            if self.time() - self.last_block_produced > \
                    BLKTIME * NUM_VALIDATORS:
                self.produce_block()
                self.last_block_produced = self.time()

    # Calculate the validator's own clock based on the actual time
    # plus a time offset that this validator happens to be wrong by
    # (eg. +1 second)
    def time(self):
        return real_time[0] + self.clock_offset


block_map = {}
listening_queue = {}
real_time = [0]
validators = {}


# Publish a block or a signature
def publish(obj):
    if isinstance(obj, Block):
        block_map[obj.hash] = obj
    # For every validator, add it to the validator's listening queue
    # at a time randomly sampled from the latency distribution
    for v in validators:
        arrival_time = real_time[0] + latency_distribution_sample()
        if arrival_time not in listening_queue:
            listening_queue[arrival_time] = []
        listening_queue[arrival_time].append((v, obj))


# One round of the clock ticking
def tick():
    for _, v in validators.items():
        v.tick()
    if real_time[0] in listening_queue:
        for validator_id, obj in listening_queue[real_time[0]]:
            validators[validator_id].on_receive(obj)
    real_time[0] += 1
    print real_time[0]


# Main function: run(7000) = simulate casper for 7000 ticks
def run(steps):
    for k in block_map.keys():
        del block_map[k]
    for k in listening_queue.keys():
        del listening_queue[k]
    for k in validators.keys():
        del validators[k]
    real_time[0] = 0
    ids[0] = 0
    for i in range(NUM_VALIDATORS):
        v = Validator()
        validators[v.id] = v
    for i in range(steps):
        tick()
    c = []
    for _, v in validators.items():
        for i, b in enumerate(v.gt.confirmed):
            assert block_map[b].height == i
        if v.gt.confirmed[:len(c)] != c[:len(v.gt.confirmed)]:
            for i in range(min(len(c), len(v.gt.confirmed))):
                if c[i] != v.gt.confirmed[i]:
                    print i, c[i], v.gt.confirmed[i]
            raise Exception("Confirmed block list mismatch")
        c.extend(v.gt.confirmed[len(c):])
    print c
