import copy, random, hashlib
from distributions import normal_distribution
import networksim
from voting_strategy import vote
import math

NUM_VALIDATORS = 20
BLKTIME = 40
BY_CHAIN = False
NETSPLITS = False

GENESIS_STATE = 0

logging_level = 0


def log(s, lvl):
    if logging_level >= lvl:
        print(s)


class Signature():
    def __init__(self, signer, probs, finalized_state, sign_from):
        self.signer = signer
        # List of maps from block hash to probability
        self.probs = probs
        # Hash of the signature (for db storage purposes)
        self.hash = random.randrange(10**14)
        # Finalized state
        self.finalized_state = finalized_state
        # Finalized height
        self.sign_from = sign_from


class Block():
    def __init__(self, maker, height):
        # The producer of the block
        self.maker = maker
        # The height of the block
        self.height = height
        # Hash of the signature (for db storage purposes)
        self.hash = random.randrange(10**20) + 10**21 + 10**23 * self.height


def state_transition(state, block):
    return state if block is None else (state ** 3 + block.hash ** 5) % 10**40


# A validator
class Validator():
    def __init__(self, pos, network):
        # Map from height to {node_id: latest_opinion}
        self.received_signatures = []
        # List of received blocks
        self.received_blocks = []
        # Own probability estimates
        self.probs = []
        # All objects that this validator has received; basically a database
        self.received_objects = {}
        self.time_received = {}
        # The validator's ID, and its position in the queue
        self.pos = self.id = pos
        # This validator's offset from the clock
        self.time_offset = normal_distribution(0, 100)()
        # The highest height that this validator has seen
        self.max_height = 0
        self.head = None
        # The last time the validator made a block
        self.last_time_made_block = -999999999999
        # The validator's hash chain
        self.finalized_hashes = []
        # Finalized states
        self.finalized_states = []
        # The highest height that the validator has finalized
        self.max_finalized_height = -1
        # The network object
        self.network = network
        # Last time signed
        self.last_time_signed = 0
        # Next neight to mine
        self.next_height = self.pos

    def get_time(self):
        return self.network.time + self.time_offset

    def sign(self, block):
        # Initialize the probability array, the core of the signature
        best_guesses = [None] * len(self.received_blocks)
        sign_from = max(0, self.max_finalized_height - 30)
        for i, b in list(enumerate(self.received_blocks))[sign_from:]:
            if self.received_blocks[i] is None:
                time_delta = self.get_time() - BLKTIME * i 
                my_opinion = 0.35 / (1 + max(0, time_delta) * 0.2 / BLKTIME) + 0.14
            else:
                time_delta = self.time_received[b.hash] - BLKTIME * i 
                my_opinion = 0.7 / (1 + abs(time_delta) * 0.2 / BLKTIME) + 0.15
                # print 'tdpost', time_delta, my_opinion
                if my_opinion == 0.5:
                    my_opinion = 0.5001
            votes = self.received_signatures[i].values() if i < len(self.received_signatures) else []
            votes += [my_opinion] * (NUM_VALIDATORS - len(votes))
            best_guesses[i] = min(vote(votes), 1 if self.received_blocks[i] is not None else my_opinion)
            if best_guesses[i] > 0.9999:
                while len(self.finalized_hashes) <= i:
                    self.finalized_hashes.append(None)
                self.finalized_hashes[i] = self.received_blocks[i].hash
            elif best_guesses[i] < 0.0001:    
                while len(self.finalized_hashes) <= i:
                    self.finalized_hashes.append(None)
                self.finalized_hashes[i] = False
        while self.max_finalized_height < len(self.finalized_hashes) - 1 \
                and self.finalized_hashes[self.max_finalized_height + 1] is not None:
            self.max_finalized_height += 1
            last_state = self.finalized_states[-1] if len(self.finalized_states) else GENESIS_STATE
            self.finalized_states.append(state_transition(last_state, self.received_blocks[self.max_finalized_height]))
        
        self.probs = self.probs[:sign_from] + best_guesses[sign_from:]
        log('Making signature: %r' % self.probs[-10:], lvl=1)
        sign_from_state = self.finalized_states[sign_from - 1] if sign_from > 0 else GENESIS_STATE
        s = Signature(self.pos, self.probs[sign_from:], sign_from_state, sign_from)
        all_signatures.append(s)
        return s

    def on_receive(self, obj):
        # Ignore objects that we already know about
        if obj.hash in self.received_objects:
            return
        # When receiving a block
        if isinstance(obj, Block):
            log('received block: %d %d' % (obj.height, obj.hash), lvl=2)
            while len(self.received_blocks) <= obj.height:
                self.received_blocks.append(None)
            self.received_blocks[obj.height] = obj
            self.time_received[obj.hash] = self.get_time()
            # If we have not yet produced a signature at this height, do so now
            s = self.sign(obj)
            self.network.broadcast(self, s)
            self.on_receive(s)
            self.network.broadcast(self, obj)
        # When receiving a signature
        elif isinstance(obj, Signature):
            while len(self.received_signatures) <= len(obj.probs) + obj.sign_from:
                self.received_signatures.append({})
            for i, p in enumerate(obj.probs):
                self.received_signatures[i + obj.sign_from][obj.signer] = p
            self.network.broadcast(self, obj)
        # Received an object request, respond if we have it
        elif isinstance(obj, ObjRequest):
            if obj.ask_hash in self.received_objects:
                self.network.direct_send(obj.sender_id, ObjResponse(
                                         self.received_objects[obj.ask_hash]))
        # Received an object response, add to database
        elif isinstance(obj, ObjResponse):
            self.received_objects[obj.obj.hash] = obj.obj
        self.received_objects[obj.hash] = obj
        self.time_received[obj.hash] = self.get_time()

    # Run every tick
    def tick(self):
        mytime = self.get_time()
        target_time = BLKTIME * self.next_height
        if mytime >= target_time:
            o = Block(self.pos, self.next_height)
            self.next_height += NUM_VALIDATORS
            log('making block: %d %d' % (o.height, o.hash), lvl=1)
            if random.random() < 0.9:
                self.network.broadcast(self, o)
            while len(self.received_blocks) <= o.height:
                self.received_blocks.append(None)
            self.received_blocks[o.height] = o
            self.received_objects[o.hash] = o
            self.time_received[o.hash] = mytime
            return o

validator_list = []
future = {}
discarded = {}
finalized_blocks = {}
all_signatures = []
now = [0]



def who_heard_of(h, n):
   o = ''
   for x in n.agents:
       o += '1' if h in x.received_objects else '0'
   return o

def get_opinions(n):
    o = []
    maxheight = 0
    for x in n.agents:
        maxheight = max(maxheight, len(x.probs))
    for h in range(maxheight):
        p = ''
        q = ''
        for x in n.agents:
            if len(x.probs) <= h:
                p += '_'  
            elif x.probs[h] < 0.5:
                p += str(int(5 - math.log(x.probs[h]) / math.log(0.0001) * 4) if x.probs[h] > 0.0001 else 0)
            elif x.probs[h] >= 0.5:
                p += str(int(5 + math.log(1 - x.probs[h]) / math.log(0.0001) * 4) if x.probs[h] < 0.9999 else 9)
            q += 'n' if len(x.received_blocks) <= h or x.received_blocks[h] is None else 'y'
        o.append((h, p, q))
    return o

def get_finalization_heights(n):
   o = []
   for x in n.agents:
       o.append(x.max_finalized_height)
   return o


# Check how often blocks that are assigned particular probabilities of
# finalization by our algorithm are actually finalized
def calibrate(finalized_hashes):
    thresholds = [0, 0.25, 0.5, 0.75] + [1 - 0.5**k for k in range(10)] + [1]
    signed = [0] * (len(thresholds) - 1)
    _finalized = [0] * (len(thresholds) - 1)
    _discarded = [0] * (len(thresholds) - 1)
    for s in all_signatures:
        for i, prob in enumerate(s.probs):
            if i + s.sign_from >= len(finalized_hashes):
                continue
            actual_result = 1 if finalized_hashes[i + s.sign_from] else 0
            index = 0
            while prob > thresholds[index + 1]:
                index += 1
            signed[index] += 1
            if actual_result == 1:
                _finalized[index] += 1
            elif actual_result == 0:
                _discarded[index] += 1
    for i in range(len(thresholds) - 1):
        if _finalized[i] + _discarded[i]:
            print 'Probability from %f to %f: %f' % (thresholds[i], thresholds[i+1], _finalized[i] * 1.0 / (_finalized[i] + _discarded[i]))
    print 'Percentage of blocks nonempty: %f%%' % (len([x for x in finalized_hashes if x]) * 100.0 / len(finalized_hashes))


def run(steps=4000):
    n = networksim.NetworkSimulator()
    for i in range(NUM_VALIDATORS):
        n.agents.append(Validator(i, n))
    n.generate_peers()
    while len(all_signatures):
        all_signatures.pop()
    for x in future.keys():
        del future[x]
    for x in finalized_blocks.keys():
        del finalized_blocks[x]
    for x in discarded.keys():
        del discarded[x]
    for i in range(steps):
        n.tick()
        if i % 500 == 0:
            print get_opinions(n)[-60:]
            finalized0 = [(v.max_finalized_height, v.finalized_hashes) for v in n.agents]
            finalized = sorted(finalized0, key=lambda x: len(x[1]))
            for j in range(len(n.agents) - 1):
                for k in range(len(finalized[j][1])):
                    if finalized[j][1][k] is not None and finalized[j+1][1][k] is not None:
                        if finalized[j][1][k] != finalized[j+1][1][k]:
                            print finalized[j]
                            print finalized[j+1]
                            raise Exception("Finalization mismatch: %r %r" % (finalized[j][1][k], finalized[j+1][1][k]))
            print 'Finalized status: %r' % [x[0] for x in finalized0]
        if i == 10000 and NETSPLITS:
            print "###########################################################"
            print "Knocking off 20% of the network!!!!!"
            print "###########################################################"
            n.knock_offline_random(NUM_VALIDATORS // 5)
        if i == 20000 and NETSPLITS:
            print "###########################################################"
            print "Simluating a netsplit!!!!!"
            print "###########################################################"
            n.generate_peers()
            n.partition()
        if i == 30000 and NETSPLITS:
            print "###########################################################"
            print "Network health back to normal!"
            print "###########################################################"
            n.generate_peers()
    calibrate(n.agents[0].finalized_hashes[:n.agents[0].max_finalized_height + 1])
