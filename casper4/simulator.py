# Implements Minimal Slashing Conditions and dynamic validator sets, descriptions here:
# Slashing Conditions: https://docs.google.com/document/d/1ecFPYhe7YsKNQUAx48S8hoyK9Y4Rbe9be_lCe_vj2ek
# Dynamic Validator Sets: https://medium.com/@VitalikButerin/safety-under-dynamic-validator-sets-ef0c3bbdf9f6#.igylifcm9

import random

POOL_SIZE = 10
VALIDATOR_IDS = range(0, POOL_SIZE*2)
INITIAL_VALIDATORS = range(0, POOL_SIZE)
BLOCK_TIME = 100
EPOCH_LENGTH = 5
AVG_LATENCY = 255

def poisson_latency(latency):
    return lambda: 1 + int(random.gammavariate(1, 1) * latency)

class Network():
    def __init__(self, latency):
        self.nodes = []
        self.latency = latency
        self.time = 0
        self.msg_arrivals = {}

    def broadcast(self, msg):
        for i, n in enumerate(self.nodes):
            delay = self.latency()
            if self.time + delay not in self.msg_arrivals:
                self.msg_arrivals[self.time + delay] = []
            self.msg_arrivals[self.time + delay].append((i, msg))

    def tick(self):
        if self.time in self.msg_arrivals:
            for node_index, msg in self.msg_arrivals[self.time]:
                self.nodes[node_index].on_receive(msg)
            del self.msg_arrivals[self.time]
        for n in self.nodes:
            n.tick(self.time)
        self.time += 1

class Block():
    def __init__(self, parent=None, finalized_dynasties=None):
        self.hash = random.randrange(10**30)
        # If we are genesis block, set initial values
        if not parent:
            self.number = 0
            self.prevhash = 0
            self.prev_dynasty = self.current_dynasty = Dynasty(INITIAL_VALIDATORS)
            self.next_dynasty = self.generate_next_dynasty(self.current_dynasty.number)
            return
        # Set our block number and our prevhash
        self.number = parent.number + 1
        self.prevhash = parent.hash
        # Generate a random next dynasty
        self.next_dynasty = self.generate_next_dynasty(parent.current_dynasty.number)
        # If the current_dynasty was finalized, we advance to the next dynasty
        if parent.current_dynasty in finalized_dynasties:
            self.prev_dynasty = parent.current_dynasty
            self.current_dynasty = parent.next_dynasty
            return
        # `current_dynasty` has not yet been finalized so we don't rotate validators
        self.prev_dynasty = parent.prev_dynasty
        self.current_dynasty = parent.current_dynasty

    @property
    def epoch(self):
        return self.number // EPOCH_LENGTH

    def generate_next_dynasty(self, prev_dynasty_number):
        random.seed(self.hash)
        next_dynasty = Dynasty(random.sample(VALIDATOR_IDS, POOL_SIZE), prev_dynasty_number+1)
        random.seed()
        return next_dynasty

class Prepare():
    def __init__(self, view, _hash, view_source, sender):
        self.view = view
        self.hash = random.randrange(10**30)
        self.blockhash = _hash
        self.view_source = view_source
        self.sender = sender

class Commit():
    def __init__(self, view, _hash, sender):
        self.view = view
        self.hash = random.randrange(10**30)
        self.blockhash = _hash
        self.sender = sender

class Dynasty():
    def __init__(self, validators, number=0):
        self.validators = validators
        self.number = number

    def __hash__(self):
        return hash(str(self.number) + str(self.validators))

    def __eq__(self, other):
        return (str(self.number) + str(self.validators)) == (str(other.number) + str(other.validators))

GENESIS = Block()

# Fork choice rule:
# 1. HEAD = genesis
# 2. Find the descendant with the highest number of commits
# 3. Repeat 2 until 0 commits
# 4. Longest chain rule

class Node():
    def __init__(self, network, id):
        # List of highest-commit descendants along with their commit counts, in oldest-to-newest order
        self.checkpoints = [GENESIS.hash]
        # Received blocks
        self.received = {GENESIS.hash: GENESIS}
        # Messages that will be processed once a given message is received
        self.dependencies = {}
        # Checkpoint to view source to prepare count
        self.prepare_count = {}
        # Checkpoints that can be committed
        self.committable = {}
        # Commits for any given checkpoint
        # Genesis is an immutable start of the chain
        self.commits = {GENESIS.hash: INITIAL_VALIDATORS}
        # Set of finalized dynasties
        self.finalized_dynasties = set()
        self.finalized_dynasties.add(Dynasty(INITIAL_VALIDATORS))
        # My current epoch
        self.current_epoch = 0
        # My highest committed epoch and hash
        self.highest_committed_epoch = -1
        self.highest_committed_hash = GENESIS.hash
        # Network I am connected to
        self.network = network
        network.nodes.append(self)
        # Longest tail from each checkpoint
        self.tails = {GENESIS.hash: GENESIS}
        # Tail that each block belongs to
        self.tail_membership = {GENESIS.hash: GENESIS.hash}
        # This node's ID
        self.id = id

    @property
    def head(self):
        latest_checkpoint = self.checkpoints[-1]
        latest_block = self.tails[latest_checkpoint]
        return latest_block

    # Get the checkpoint immediately before a given checkpoint
    def get_checkpoint_parent(self, block):
        if block.number == 0:
            return None
        return self.received[self.tail_membership[block.prevhash]]

    # If we received an object but did not receive some dependencies
    # needed to process it, save it to be processed later
    def add_dependency(self, _hash, obj):
        if _hash not in self.dependencies:
            self.dependencies[_hash] = []
        self.dependencies[_hash].append(obj)

    # Is a given checkpoint an ancestor of another given checkpoint?
    def is_ancestor(self, anc, desc):
        if not isinstance(anc, Block):
            anc = self.received[anc]
        if not isinstance(desc, Block):
            desc = self.received[desc]
        assert anc.number % EPOCH_LENGTH == 0
        assert desc.number % EPOCH_LENGTH == 0
        while True:
            if desc is None:
                return False
            if desc.hash == anc.hash:
                return True
            desc = self.get_checkpoint_parent(desc)

    def get_last_committed_checkpoint(self):
        z = len(self.checkpoints) - 1
        while self.score_checkpoint(self.received[self.checkpoints[z]]) < 1:
            z -= 1
        return self.checkpoints[z]

    # Called on receiving a block
    def accept_block(self, block):
        # If we didn't receive the block's parent yet, wait
        if block.prevhash not in self.received:
            self.add_dependency(block.prevhash, block)
            return False
        # We recived the block
        self.received[block.hash] = block
        # print(self.id, 'got a block', block.number, block.hash)
        # If it's an epoch block (in general)
        if block.number % EPOCH_LENGTH == 0:
            #  Start a tail object for it
            self.tail_membership[block.hash] = block.hash
            self.tails[block.hash] = block
        # Otherwise...
        else:
            # See if it's part of the longest tail, if so set the tail accordingly
            assert block.prevhash in self.received
            assert block.prevhash in self.tail_membership
            self.tail_membership[block.hash] = self.tail_membership[block.prevhash]
            if block.number > self.tails[self.tail_membership[block.hash]].number:
                self.tails[self.tail_membership[block.hash]] = block
        self.check_checkpoints(self.received[self.tail_membership[block.hash]])
        self.maybe_prepare_last_checkpoint()
        return True

    def maybe_prepare_last_checkpoint(self):
        target_block = self.received[self.checkpoints[-1]]
        # If the block is an epoch block of a higher epoch than what we've seen so far
        if target_block.epoch > self.current_epoch:
            print('now in epoch %d' % target_block.epoch)
            # Increment our epoch
            self.current_epoch = target_block.epoch
            # If our highest committed hash is in the main chain (in most cases
            # it should be), then send a prepare
            last_committed_checkpoint = self.get_last_committed_checkpoint()
            if self.is_ancestor(self.highest_committed_hash, last_committed_checkpoint):
                print('Preparing %d for epoch %d with view source %d' %
                      (target_block.hash, target_block.epoch, self.received[last_committed_checkpoint].epoch))
                self.network.broadcast(Prepare(target_block.epoch, target_block.hash, self.received[last_committed_checkpoint].epoch, self.id))
                assert self.received[target_block.hash]

    # Pick a checkpoint by number of commits first, epoch number
    # (ie. longest chain rule) second
    def score_checkpoint(self, block):
        # Choose the dynasty (current or previous) with the minimum number of commits
        current_dynasty_number_of_commits = len(list(set(block.current_dynasty.validators) & set(self.commits.get(block.hash, []))))
        prev_dynasty_number_of_commits = len(list(set(block.prev_dynasty.validators) & set(self.commits.get(block.hash, []))))
        number_of_commits = min(current_dynasty_number_of_commits, prev_dynasty_number_of_commits)
        return number_of_commits + 0.000000001 * self.tails[block.hash].number

    # See if a given epoch block requires us to reorganize our checkpoint list
    def check_checkpoints(self, block):
        # Is this hash already in our main chain? Then do nothing
        if block.hash in self.checkpoints:
            # prev_checkpoint = self.received[self.checkpoints[self.checkpoints.index(block.hash) - 1]]
            # if score_checkpoint(block) < score_checkpoint(prev_checkpoint):
            return
        # Figure out how many of our checkpoints we need to revert
        z = len(self.checkpoints) - 1
        new_score = self.score_checkpoint(block)
        while new_score > self.score_checkpoint(self.received[self.checkpoints[z]]):
            z -= 1
        # If none, do nothing
        if z == len(self.checkpoints) - 1 and block.number <= self.received[self.checkpoints[z-1]].number:
            return
        # Delete the checkpoints that need to be superseded
        self.checkpoints = self.checkpoints[:z + 1]
        # Re-run the fork choice rule
        while 1:
            # Find the descendant with the highest score (commits first, epoch second)
            max_score = 0
            max_descendant = None
            for _hash in self.tails:
                if self.is_ancestor(self.checkpoints[-1], _hash) and _hash != self.checkpoints[-1]:
                    new_score = self.score_checkpoint(self.received[_hash])
                    if new_score > max_score:
                        max_score = new_score
                        max_descendant = _hash
            # Append to the chain that checkpoint, and all checkpoints between the
            # last checkpoint and the new one
            if max_descendant:
                new_chain = [max_descendant]
                while new_chain[0] != self.checkpoints[-1]:
                    new_chain.insert(0, self.get_checkpoint_parent(self.received[new_chain[0]]).hash)
                self.checkpoints.extend(new_chain[1:])
            # If there were no suitable descendants found, break
            else:
                break
        print('New checkpoints: %r' % [self.received[b].epoch for b in self.checkpoints])

    # Called on receiving a prepare message
    def accept_prepare(self, prepare):
        if self.id == 0:
            print('got a prepare', prepare.view, prepare.view_source, prepare.blockhash, prepare.blockhash in self.received)
        # If the block has not yet been received, wait
        if prepare.blockhash not in self.received:
            self.add_dependency(prepare.blockhash, prepare)
            return False
        # If the sender is not in the prepare's dynasty, ignore the prepare
        if prepare.sender not in self.received[prepare.blockhash].current_dynasty.validators and \
                prepare.sender not in self.received[prepare.blockhash].prev_dynasty.validators:
            return False
        # Add to the prepare count
        if prepare.blockhash not in self.prepare_count:
            self.prepare_count[prepare.blockhash] = {}
        self.prepare_count[prepare.blockhash][prepare.view_source] = self.prepare_count[prepare.blockhash].get(prepare.view_source, 0) + 1
        # If there are enough prepares and the previous dynasty is finalized...
        if self.prepare_count[prepare.blockhash][prepare.view_source] > (POOL_SIZE * 2) // 3 and \
                self.received[prepare.blockhash].prev_dynasty in self.finalized_dynasties and \
                prepare.blockhash not in self.committable:
            # Mark it as committable
            self.committable[prepare.blockhash] = True
            # Start counting commits
            self.commits[prepare.blockhash] = []
            # If there are dependencies (ie. commits that arrived before there
            # were enough prepares), since there are now enough prepares we
            # can process them
            if "commit:"+str(prepare.blockhash) in self.dependencies:
                for c in self.dependencies["commit:"+str(prepare.blockhash)]:
                    self.accept_commit(c)
                del self.dependencies["commit:"+str(prepare.blockhash)]
            # Broadcast a commit
            if self.current_epoch == prepare.view:
                self.network.broadcast(Commit(prepare.view, prepare.blockhash, self.id))
                print('Committing %d for epoch %d' % (prepare.blockhash, prepare.view))
                self.highest_committed_epoch = prepare.view
                self.highest_committed_hash = prepare.blockhash
                self.current_epoch = prepare.view + 0.5
        return True

    # Called on receiving a commit message
    def accept_commit(self, commit):
        if self.id == 0:
            print('got a commmit', commit.view, commit.blockhash, commit.blockhash in self.received, commit.blockhash in self.committable)
        # If the block has not yet been received, wait
        if commit.blockhash not in self.received:
            self.add_dependency(commit.blockhash, commit)
            return False
        # If the sender is not in the commit's dynasty, ignore the commit
        if commit.sender not in self.received[commit.blockhash].current_dynasty.validators and \
                commit.sender not in self.received[commit.blockhash].prev_dynasty.validators:
            return False
        # If there have not yet been enough prepares, wait
        if commit.blockhash not in self.committable:
            self.add_dependency("commit:"+str(commit.blockhash), commit)
            return False
        # Add the commit by recording the sender
        self.commits[commit.blockhash].append(commit.sender)
        # Check if the block is finalized
        current_dynasty_commits = list(set(self.received[commit.blockhash].current_dynasty.validators) & set(self.commits[commit.blockhash]))
        prev_dynasty_commits = list(set(self.received[commit.blockhash].prev_dynasty.validators) & set(self.commits[commit.blockhash]))
        if len(current_dynasty_commits) > (POOL_SIZE * 2) // 3 and len(prev_dynasty_commits) > (POOL_SIZE * 2) // 3:
            # Because the block has been finalized let's record its dynasty as finalized
            finalized_dynasty = self.received[commit.blockhash].current_dynasty
            self.finalized_dynasties.add(finalized_dynasty)
            print('Finalizing dynasty number %d for block number %d' %
                  (finalized_dynasty.number, self.received[commit.blockhash].number))
        # Update the checkpoints if needed
        self.check_checkpoints(self.received[commit.blockhash])
        return True

    # Called on receiving any object
    def on_receive(self, obj):
        if obj.hash in self.received:
            return False
        if isinstance(obj, Block):
            o = self.accept_block(obj)
        elif isinstance(obj, Prepare):
            o = self.accept_prepare(obj)
        elif isinstance(obj, Commit):
            o = self.accept_commit(obj)
        # If the object was successfully processed
        # (ie. not flagged as having unsatisfied dependencies)
        if o:
            self.received[obj.hash] = obj
            if obj.hash in self.dependencies:
                for d in self.dependencies[obj.hash]:
                    self.on_receive(d)
                del self.dependencies[obj.hash]

    # Called every round
    def tick(self, _time):
        if self.id == (_time // BLOCK_TIME) % POOL_SIZE and _time % BLOCK_TIME == 0:
            new_block = Block(self.head, self.finalized_dynasties)
            self.network.broadcast(new_block)
            self.on_receive(new_block)

network = Network(poisson_latency(AVG_LATENCY))
nodes = [Node(network, i) for i in VALIDATOR_IDS]
for t in range(25000):
    network.tick()
    if t % 1000 == 999:
        print('Heads:', [n.head.number for n in nodes])
        print('Checkpoints:', nodes[0].checkpoints)
        print('Commits:', [nodes[0].commits.get(c, 0) for c in nodes[0].checkpoints])
        print('Blocks Dynasties:', [(nodes[0].received[c].current_dynasty.number) for c in nodes[0].checkpoints])
        print('All Node Dynasties:', [(node.tails[node.checkpoints[-1]].current_dynasty.number) for node in nodes])
