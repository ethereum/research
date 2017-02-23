# NOTE: This script is totally untested and likely has many bugs.
# Be warned!
import random

NODE_COUNT = 20
BLOCK_TIME = 100
EPOCH_LENGTH = 100
AVG_LATENCY = 30

def poisson_latency(latency):
    return lambda: int(random.gammavariate(1, 1) * latency)

class Network():
    def __init__(self, latency):
        self.nodes = []
        self.latency = latency
        self.time = 0
        self.msg_arrivals = {}

    def broadcast(self, msg):
        delay = self.latency()
        for i, n in enumerate(self.nodes):
            if self.time + delay not in self.msg_arrivals[i]:
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
    def __init__(self, parent=None):
        if not parent:
            self.number = 0
            self.prevhash = 0
        else:    
            self.number = parent.number + 1
            self.prevhash = parent.hash
        self.hash = random.randrange(10**30)

    @property
    def epoch(self):
        return self.number // EPOCH_LENGTH

class Prepare():
    def __init__(self, view, _hash, view_source):
        self.view = view
        self.hash = _hash
        self.view_source = view_source

class Commit():
    def __init__(self, view, _hash):
        self.view = view
        self.hash = _hash

GENESIS = Block()

# Fork choice rule:
# 1. HEAD = genesis
# 2. Find the descendant with the highest number of commits
# 3. Repeat 2 until 0 commits
# 4. Longest chain rule

class Node():
    def __init__(self, network, id):
        # List of highest-commit descendants along with their commit counts, in oldest-to-newest order
        self.checkpoints = [GENESIS]
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
        self.commits = {GENESIS.hash: 101}
        # My current epoch
        self.current_epoch = 0
        # My highest committed epoch and hash
        self.highest_committed_epoch = -1
        self.highest_committed_hash = None
        # Network I am connected to
        self.network = network
        network.nodes.append(self)
        # Longest tail from each checkpoint
        self.tails = {}
        # Tail that each block belongs to
        self.tail_membership = {}
        # This node's ID
        self.id = id

    # Get the checkpoint immediately before a given checkpoint
    def get_checkpoint_parent(self, block):
        if block.number == 0:
            return None
        return self.receives[self.tail_membership[block.prevhash]]

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

    # Called on receiving a block
    def accept_block(self, block):
        # If we didn't receive the block's parent yet, wait
        if block.prevhash not in self.received:
            self.add_dependency(block.prevhash, block)
            return
        # We recived the block
        self.received[block.hash] = block
        # If the block is an epoch block of a higher epoch
        if block.number == (self.current_epoch + 1) * EPOCH_LENGTH:
            # Increment our epoch
            self.current_epoch = block.epoch
            # If our highest committed hash is in the main chain (in most cases
            # it should be), then send a prepare
            if self.is_ancestor(self.highest_committed_hash, self.checkpoints[-1]):
                self.network.broadcast(Prepare(self.current_epoch, block.hash, self.received[self.checkpoints[-1]].epoch))
        # If it's an epoch block (in general)
        if block.number % EPOCH_LENGTH == 0:
            #  Start a tail object for it
            self.tail_membership[block.hash] = block.hash
            self.tails[block.hash] = block
        # Otherwise...
        else:
            # See if it's part of the longest tail, if so set the tail accordingly
            self.tail_membership[block.hash] = self.tail_membership[block.prevhash]
            if block.number > self.tails[self.tail_membership[block.hash]].number:
                self.tails[self.tail_membership[block.hash]] = block

    # Pick a checkpoint by number of commits first, epoch number
    # (ie. longest chain rule) second
    def score_checkpoint(self, block):
        return self.commits.get(block, 0) + 0.000000001 * block.number

    # See if a given epoch block requires us to reorganize our checkpoint list
    def check_checkpoints(self, block):
        # Is this hash already in our main chain? Then do nothing
        if block.hash in self.checkpoints:
            # prev_checkpoint = self.received[self.checkpoints[self.checkpoints.index(block.hash) - 1]]
            # if score_checkpoint(block) < score_checkpoint(prev_checkpoint):
            return
        # Figure out how many of our checkpoints we need to revert
        z = len(self.checkpoints - 1)
        new_score = score_checkpoint(block)
        while new_score > score_checkpoint(self.checkpoints[z]):
            z -= 1
        # If none, do nothing
        if z == len(self.checkpoints - 1):
            return
        # Delete the checkpoints that need to be superseded
        self.checkpoints = self.checkpoints[:z + 1]
        # Re-run the fork choice rule
        while 1:
            # Find the descendant with the highest score (commits first, epoch second)
            max_score = 0
            max_descendant = None
            for _hash in self.descendants:
                if self.is_ancestor(self.checkpoints[z], _hash):
                    new_score = score_checkpoint(self.received[_hash])
                    if new_score > max_score:
                        max_score = new_score
                        max_descendant = _hash
            # Append to the chain that checkpoint, and all checkpoints between the
            # last checkpoint and the new one
            if max_descendant:
                new_chain = [max_descendant.hash]
                while new_chain[0].hash != self.checkpoints[z].hash:
                    new_chain.insert(0, self.get_checkpoint_parent(new_chain[0]))
                self.checkpoints.append(new_chain[1:])
            # If there were no suitable descendants found, break
            else:
                break
    
    # Called on receiving a prepare message
    def accept_prepare(self, prepare):
        # If the block has not yet been received, wait
        if prepare.hash not in self.received:
            self.add_dependency(prepare.hash, prepare)
            return
        # Add to the prepare count
        if prepare.hash not in self.prepare_count:
            self.prepare_count[prepare.hash] = {}
        self.prepare_count[prepare.hash][prepare.view_source] = self.prepare_count[prepare.hash].get(prepare.view_source, 0) + 1
        # If there are enough prepares...
        if self.prepare_count[prepare.hash][prepare.view_source] > (NODE_COUNT * 2) // 3:
            # Mark it as committable
            self.committable[prepare.hash] = True
            # Start counting commits
            self.commits[prepare.hash] = 0
            # If there are dependencies (ie. commits that arrived before there
            # were enough prepares), since there are now enough prepares we
            # can process them
            if "commit:"+str(prepare.hash) in self.dependencies:
                for c in self.dependencies["commit:"+str(prepare.hash)]:
                    self.accept_commit(c)
                del self.dependencies["commit:"+str(prepare.hash)]
            # Broadcast a commit
            if self.current_epoch == prepare.view:
                self.network.broadcast(Commit(prepare.view, prepare.hash))
                self.highest_committed_epoch = prepare.view
                self.highest_committed_hash = prepare.hash
                self.current_epoch = prepare.view + 1

    # Called on receiving a commit message
    def accept_commit(self, commit):
        # If the block has not yet been received, wait
        if commit.hash not in self.received:
            self.add_dependency(commit.hash, commit)
            return
        # If there have not yet been enough prepares, wait
        if commit.hash not in self.committable:
            self.add_dependency("commit:"+str(commit.hash), commit)
            return
        # Add commits, and update checkpoints if needed
        self.commits[commit.hash] += 1
        if self.commits[commit.hash] % 10 == 0:
            self.check_checkpoints(self.received[commit.hash])

    # Called on receiving any object
    def on_receive(self, obj):
        if isinstance(obj, Block):
            self.accept_block(obj)
        elif isinstance(obj, Prepare):
            self.accept_prepare(obj)
        elif isinstance(obj, Commit):
            self.accept_commit(obj)
        self.received[obj.hash] = obj
        if obj.hash in self.dependencies:
            for d in self.dependencies[obj.hash]:
                self.on_receive(d)
            del self.dependencies[obj.hash]

    # Called every round
    def tick(self):
        if self.id == (self.time // BLOCK_TIME) % NODE_COUNT and self.time % BLOCK_TIME == 0:
            latest_checkpoint = self.checkpoints[-1]
            latest_block = self.tails[latest_checkpoint]
            new_block = Block(latest_block)
            self.network.broadcast(new_block)

n = Network(poisson_latency(AVG_LATENCY))
nodes = [Node(n, i) for i in range(NODE_COUNT)]
for t in range(10000):
    n.tick()
