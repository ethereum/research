import copy, random, hashlib
from distributions import normal_distribution
import networksim
from voting_strategy import vote

NUM_VALIDATORS = 20
BLKTIME = 100

logging_level = 0


def log(s, lvl):
    if logging_level >= lvl:
        print(s)


class Signature():
    def __init__(self, signer, probs, height, last_finalized):
        self.signer = signer
        # List of maps from block hash to probability
        self.probs = probs
        # Top height of the signature
        self.height = height
        # The HashChainObj that represents the last finalized hash
        # of this signature
        self.last_finalized = last_finalized
        # Hash of the signature (for db storage purposes)
        self.hash = random.randrange(10**14) + 10**14 * self.height


class Block():
    def __init__(self, maker, height):
        # The producer of the block
        self.maker = maker
        # The height of the block
        self.height = height
        # Hash of the signature (for db storage purposes)
        self.hash = random.randrange(10**20) + 10**20 * self.height


# An object containing the hash of a block and the hash of a previous
# hash chain object. These objects end up forming the actual "blockchain"
# in this scheme
class HashChainObj():
    def __init__(self, block, prev):
        # Genesis hash chain object
        if block is None:
            self.hash = 0
            self.prev = None
            self.blockhash = None
        # All other hash chain objects
        else:
            self.hash = (prev.hash ** 7 + block.hash ** 3) ** 5 % 10**30
            self.prev = prev.hash
            self.blockhash = block.hash


# A request for info from a node that needs to synchronize after a period
# of being offline from the network
class SyncRequest():
    def __init__(self, sender):
        self.sender_id = sender.id
        self.last_finalized_height = sender.max_finalized_height
        self.hash = random.randrange(10**10)


# A response to a sync request
class SyncResponse():
    def __init__(self, blocks, signatures, finalized_chain, requester_mfh, responder_mfh, responder_hashchain):
        self.blocks = blocks
        self.signatures = signatures
        self.finalized_chain = finalized_chain
        self.requester_mfh = requester_mfh
        self.responder_mfh = responder_mfh
        self.responder_hashchain = responder_hashchain
        self.hash = random.randrange(10**10)


# A request to get an object with a particular hash
class ObjRequest():
    def __init__(self, sender, ask_hash):
        self.sender_id = sender.id
        self.ask_hash = ask_hash
        self.hash = random.randrange(10**10)


# A response to an object request
class ObjResponse():
    def __init__(self, obj):
        self.obj = obj
        self.hash = random.randrange(10**10)


# A validator
class Validator():
    def __init__(self, pos, network):
        # Map from height to { blocks: blocks, signatures: signatures }
        self.heights = {0: {"blocks": {}, "signatures": {}}}
        # All objects that this validator has received; basically a database
        self.received_objects = {}
        # The validator's ID, and its position in the queue
        self.pos = self.id = pos
        # Heights that this validator has already signed
        self.signed_heights = {}
        # This validator's offset from the clock
        self.time_offset = max(normal_distribution(200, 100)(), 0)
        # The highest height that this validator has seen
        self.max_height = 0
        self.head = None
        # The last time the validator made a block
        self.last_time_made_block = -999999999999
        # Block hashes of finalized blocks
        self.finalized = [None]
        # The validator's hash chain
        self.finalized_hashes = [HashChainObj(None, None)]
        # The highest height that the validator has finalized
        self.max_finalized_height = 0
        # The network object
        self.network = network

    def sign(self, block):
        # Calculate this validator's (randomly offset) view of the current time
        mytime = now[0] + self.time_offset
        offset = (mytime - (block.maker * BLKTIME)) % (BLKTIME * NUM_VALIDATORS)
        # Iniitalize the probability array, the core of the signature
        probs = []
        # Compute the validator's opinion of the latest block, based on whether
        # or not it arrived at the correct time
        if offset < BLKTIME:
            probs.append({block.hash: 0.67 + random.random() * 0.05})
        else:
            probs.append({block.hash: 0.33 - random.random() * 0.05})
        # Compute the validator's current view of previous blocks up to the
        # point of finalization
        probs.extend(self.compute_view(block.height))
        if self.pos == 0:
            for i, v in enumerate(probs):
                log('Signatures for block %d: %r' % (block.height - i, v), lvl=1)
        # Add the end of the node's finalized hash chain to the signature, and
        # create the signature
        pre_probs_h = block.height - len(probs)
        o = Signature(self.pos, probs, block.height, self.finalized_hashes[pre_probs_h])
        # Sanity check
        if o.last_finalized.blockhash is not None:
            assert o.last_finalized.blockhash // 10**20 == block.height - len(probs)
        # Append the signature to the node's list of signatures produced and return it
        signatures.append(o)
        return o

    def compute_view(self, from_height):
        # Fetch the latest signature from each validator, along with its age
        signatures = {}
        for v in range(NUM_VALIDATORS):
            for q in xrange(1, from_height):
                if from_height - q in self.heights and v in self.heights[from_height - q]['signatures']:
                    signatures[v] = (q, self.heights[from_height - q]['signatures'][v])
                    break
        # If we have no signatures, then we have no opinion
        if len(signatures) == 0:
            return []
        # For every height between the node's current maximum seen height and
        # its last known finalized height...
        probs = []
        for i in xrange(1, from_height - self.max_finalized_height):
            block_scores = {}
            if from_height - i not in self.heights:
                self.heights[from_height - i] = {"blocks": {}, "signatures": {}}
            # For every signature...
            for q, sig in signatures.values():
                assert sig.height == from_height - q
                # If the signature is older than this height, then it has
                # nothing to say about this height so its vote for every block
                # is assumed to be zero
                if i < q:
                    continue
                # Otherwise, grab the signature's probability estimate for
                # every block at the height
                elif len(sig.probs) > i-q:
                    for blockhash, prob in sig.probs[i-q].items():
                        assert blockhash // 10**20 == from_height - q - (i-q)
                        if blockhash not in block_scores:
                            block_scores[blockhash] = []
                        block_scores[blockhash].append(prob)
                # Every signature has a hash chain object at the end; this
                # implicitly attests with probability 0.999999 that every block
                # in that chain going back is final. Hence, we go back through
                # the chain and add such an attestation to our list of votes
                else:
                    h = sig.last_finalized
                    assert h.blockhash // 10**20 == from_height - q - len(sig.probs)
                    success = True
                    for _ in range(i-q - len(sig.probs)):
                        if h.prev not in self.received_objects:
                            success = False
                            break
                        h = self.received_objects[h.prev]
                    if success and h.blockhash is not None:
                        assert h.blockhash // 10**20 == from_height - i, (h.blockhash, from_height - i)
                        if h.blockhash not in block_scores:
                            block_scores[h.blockhash] = []
                        block_scores[h.blockhash].append(0.999999)
                        if i-q-len(sig.probs) > 0:
                            log('Decoding hash succeeded', lvl=3)
                    else:
                        log('Decoding hash failed', lvl=3)
                        self.network.broadcast(self, SyncRequest(self))
            # Use the array of previous votes that we have collected, and
            # compute from that our own vote for every block
            probs.append(vote(block_scores, self.received_objects, NUM_VALIDATORS))
            for b in block_scores:
                if b not in self.received_objects:
                    self.network.broadcast(self, ObjRequest(self, b))
            # Log a single node's viewpoint changing over time
            if self.pos == 0:
                log('%d %r' % (from_height - i, self.heights[from_height - i]["blocks"].keys()), lvl=2)
                log(block_scores, lvl=2)
                log(probs[-1], lvl=2)
            # See if our vote corresponds to finality anywhere
            for blockhash, p in probs[-1].items():
                assert blockhash // 10**20 == from_height - i
                # 0.9999 = finality threshold
                if p > 0.9999:
                    while len(self.finalized) <= from_height - i:
                        self.finalized.append(None)
                    self.finalized[from_height - i] = blockhash
                    # Add the hash to a global list of finalized blocks
                    finalized_blocks[blockhash] = True
                    # Add all other hashes at that height to a global list of
                    # discarded blocks
                    if from_height - i in self.heights:
                        for b in self.heights[from_height - i]['blocks']:
                            if b != blockhash:
                                discarded[b] = True
                    # Advance the max_finalized_height and re-calculate the
                    # hash chain
                    while self.max_finalized_height + 1 < len(self.finalized) and self.finalized[self.max_finalized_height + 1] is not None:
                        self.max_finalized_height += 1
                        last_finalized_block = self.received_objects[self.finalized[self.max_finalized_height]]
                        new_state = HashChainObj(last_finalized_block, self.finalized_hashes[-1])
                        self.received_objects[new_state.hash] = new_state
                        self.finalized_hashes.append(new_state)
        # Sanity check
        for j, p in enumerate(probs):
            for h, sig in p.items():
                assert h // 10**20 == from_height - j - 1, (probs, from_height, block_scores)
        log('Probabilities: %r' % probs, lvl=4)
        return probs

    def on_receive(self, obj):
        # Ignore objects that we already know about
        if obj.hash in self.received_objects:
            return
        # When receiving a block
        if isinstance(obj, Block):
            log('received block: %d %d' % (obj.height, obj.hash), lvl=2)
            if obj.height > self.max_finalized_height + 40:
                self.network.broadcast(self, SyncRequest(self))
            # If we have not yet produced a signature at this height, do so now
            if obj.height not in self.signed_heights:
                s = self.sign(obj)
                self.signed_heights[obj.height] = True
                self.on_receive(s)
                self.network.broadcast(self, s)
            if obj.height not in self.heights:
                self.heights[obj.height] = {"blocks": {}, "signatures": {}}
            self.heights[obj.height]["blocks"][obj.hash] = obj
            if obj.height > self.max_height:
                self.max_height = obj.height
                self.head = obj
            self.network.broadcast(self, obj)
        # When receiving a signature
        elif isinstance(obj, Signature):
            if obj.height not in self.heights:
                self.heights[obj.height] = {"blocks": {}, "signatures": {}}
            self.heights[obj.height]["signatures"][obj.signer] = obj
            self.network.broadcast(self, obj)
            self.received_objects[obj.last_finalized.hash] = obj.last_finalized
        # Received a synchronization request from another node
        elif isinstance(obj, SyncRequest):
            blocks, signatures, hashchainobjs = [], [], []
            # Respond only if we have something to say
            if self.max_finalized_height > obj.last_finalized_height:
                # Add blocks and signatures at all requested and possible
                # heights
                for h in range(obj.last_finalized_height, self.max_height):
                    if h in self.heights:
                        for b in self.heights[h]["blocks"].values():
                            blocks.append(b)
                        for s in self.heights[h]["signatures"].values():
                            signatures.append(s)
                # Add the finalized hash chain
                for h in range(obj.last_finalized_height, self.max_finalized_height + 1):
                    hashchainobjs.append(self.finalized_hashes[h])
                log('Responding to request with height %d, my finalized height %d and my max height %d' % 
                    (obj.last_finalized_height, self.max_finalized_height, self.max_height), lvl=2)
                # Create and send a synchronization response object
                self.network.direct_send(obj.sender_id, SyncResponse(blocks, signatures, hashchainobjs,
                                         obj.last_finalized_height, self.max_finalized_height, self.finalized))
        # Received a synchronization response object from another node
        elif isinstance(obj, SyncResponse):
            # Process it only if the object has something to give us
            if obj.responder_mfh > self.max_finalized_height:
                log('Received response, my finalized height was %d and my height was %d, their MFH was %d'
                    % (self.max_finalized_height, self.max_height, obj.responder_mfh), lvl=2)
                for s in obj.finalized_chain:
                    self.received_objects[s.hash] = s
                for s in obj.signatures:
                    self.on_receive(s)
                for b in obj.blocks:
                    if b.height not in self.heights:
                        self.heights[b.height] = {"signatures": {}, "blocks": {}}
                    self.heights[b.height]["blocks"][b.hash] = b
                    self.received_objects[b.hash] = b
                self.compute_view(self.max_height)
                log('And now they are %d and %d' %
                    (self.max_finalized_height, self.max_height), lvl=2)
        # Received an object request, respond if we have it
        elif isinstance(obj, ObjRequest):
            if obj.ask_hash in self.received_objects:
                self.network.direct_send(obj.sender_id, ObjResponse(
                                         self.received_objects[obj.ask_hash]))
        # Received an object response, add to database
        elif isinstance(obj, ObjResponse):
            self.received_objects[obj.obj.hash] = obj.obj
        self.received_objects[obj.hash] = obj

    # Run every tick
    def tick(self):
        mytime = self.network.time + self.time_offset
        offset = (mytime - (self.pos * BLKTIME)) % (BLKTIME * NUM_VALIDATORS)
        if offset < BLKTIME and self.last_time_made_block < mytime - (BLKTIME * NUM_VALIDATORS / 2):
            self.last_time_made_block = mytime
            o = Block(self.pos, self.max_height + 1)
            log('making block: %d %d' % (o.height, o.hash), lvl=1)
            self.network.broadcast(self, o)
            self.received_objects[o.hash] = o
            return o

validator_list = []
future = {}
discarded = {}
finalized_blocks = {}
signatures = []
now = [0]


# Check how often blocks that are assigned particular probabilities of
# finalization by our algorithm are actually finalized
def calibrate():
    thresholds = [0, 0.25, 0.5, 0.75] + [1 - 0.5**k for k in range(10)] + [1]
    signed = [0] * (len(thresholds) - 1)
    _finalized = [0] * (len(thresholds) - 1)
    _discarded = [0] * (len(thresholds) - 1)
    for s in signatures:
        for probs in s.probs:
            for blockhash, p in probs.items():
                index = 0
                while p > thresholds[index + 1]:
                    index += 1
                signed[index] += 1
                if blockhash in finalized_blocks:
                    _finalized[index] += 1
                    assert blockhash not in discarded, blockhash
                if blockhash in discarded:
                    _discarded[index] += 1
                    assert blockhash not in finalized_blocks, blockhash
    for i in range(len(thresholds) - 1):
        if _finalized[i] + _discarded[i]:
            print 'Probability from %f to %f: %f' % (thresholds[i], thresholds[i+1], _finalized[i] * 1.0 / (_finalized[i] + _discarded[i]))


def run(steps=4000):
    n = networksim.NetworkSimulator()
    for i in range(NUM_VALIDATORS):
        n.agents.append(Validator(i, n))
    n.generate_peers()
    while len(signatures):
        signatures.pop()
    for x in future.keys():
        del future[x]
    for x in finalized_blocks.keys():
        del finalized_blocks[x]
    for x in discarded.keys():
        del discarded[x]
    for i in range(steps):
        n.tick()
        if i % 250 == 0:
            finalized = [(v.max_finalized_height, v.finalized) for v in n.agents]
            finalized = sorted(finalized, key=lambda x: len(x[1]))
            for j in range(len(n.agents) - 1):
                for k in range(len(finalized[j][1])):
                    assert finalized[j][1][k] is None or finalized[j+1][1][k] is \
                        None or finalized[j][1][k] == finalized[j+1][1][k], (finalized, j)
            print 'Finalized status: %r' % [x[1][x[0]] for x in finalized]
        if i == 10000:
            print "###########################################################"
            print "Knocking off 20% of the network!!!!!"
            print "###########################################################"
            n.knock_offline_random(NUM_VALIDATORS // 5)
        if i == 20000:
            print "###########################################################"
            print "Simluating a netsplit!!!!!"
            print "###########################################################"
            n.generate_peers()
            n.partition()
        if i == 30000:
            print "###########################################################"
            print "Network health back to normal!"
            print "###########################################################"
            n.generate_peers()
    calibrate()
