# The purpose of this script is to create an evolutionary
# model to study the equilibrium effects of Bitcoin Unlimited-style
# "emergent consensus". Note that the model is not yet quite
# complete as it does not take into account the benefits of
# mining "sister blocks" that steal transaction fees, though it
# does give a rough idea of what equilibrium behavior
# among the various miner policy dimensions (block accept size,
# override depth, block creation size) looks like
import random

# Block reward
REWARD = 1000
# Call this function to get a tx with the right fee
TX_FEE_DISTRIBUTION = lambda: (10000 // random.randrange(5, 250)) * 0.01
# TX_FEE_DISTRIBUTION = lambda: 20
# Propagation time
PROPTIME_FACTOR = 1

# List of tuples:
# (default limit, n-block limit, acceptance depth, creation limit)
strategies = []
for i in range(4):
    for j in range(4):
        strategies.append([2 + i * 2, 100, 3, 10 + j * 4])

class Block():
    def __init__(self, parent, size, fees, miner):
        self.hash = random.randrange(10**20)
        self.parent = parent
        self.score = 1 if self.parent is None else parent.score + 1
        self.miner = miner
        self.size = size
        self.fees = fees

class Miner():
    def __init__(self, strategy, id):
        self.limit, self.big_limit, self.accept_depth, self.creation_limit = strategy
        self.chain = {}
        self.big_chain = {}
        self.head = None
        self.big_head = None
        self.id = id
        self.future = {}
        self.children = {}
        self.created = 0

    def process_history(self, time):
        deletes = []
        for t in self.future:
            if t <= time:
                for b in self.future[t]:
                    self.process_block(b)
                deletes.append(t)
        for t in deletes:
            del self.future[t]

    def add_block(self, block, time):
        self.process_history(time)
        if time + int(block.size * PROPTIME_FACTOR) not in self.future:
            self.future[time + int(block.size * PROPTIME_FACTOR)] = [block]
        else:
            self.future[time + int(block.size * PROPTIME_FACTOR)].append(block)

    def process_block(self, block):
        if block.size <= self.limit and (block.parent is None or block.parent.hash in self.chain):
            self.chain[block.hash] = block
            if block.score > (self.head.score if self.head else 0):
                self.head = block
        if block.size <= self.big_limit and (block.parent is None or block.parent.hash in self.big_chain):
            self.big_chain[block.hash] = block
            if block.score > (self.big_head.score if self.big_head else 0):
                self.big_head = block
            if block.score > (self.head.score if self.head else 0) + self.accept_depth:
                self.head = block
                self.chain[block.hash] = block
        if block.parent and block.parent.hash not in self.chain and block.parent.hash not in self.big_chain:
            if block.parent.hash not in self.children:
                self.children[block.parent.hash] = [block]
            else:
                self.children[block.parent.hash].append(block)
        if block.hash in self.children:
            for c in self.children[block.hash]:
                self.process_block(c)
            del self.children[block.hash]

    def create_block(self, backlog, time):
        self.process_history(time)
        fees = sum(backlog[:self.creation_limit])
        # print 'Creating block of size %d (fees %d, seq %d)' % (self.creation_limit, fees, self.head.score + 1 if self.head else 1)
        self.created += 1
        return Block(self.head, self.creation_limit, fees, self.id)


def simulate(strats):
    miners = [Miner(strat, i) for i, strat in enumerate(strats)]
    backlog = []
    for i in range(100000):
        if i % 10000 == 0:
            print 'Progress %d' % i
        backlog.append(TX_FEE_DISTRIBUTION())
        if random.random() < 0.01:
            backlog = sorted(backlog)[::-1]
            miner = random.choice(miners)
            b = miner.create_block(backlog, i)
            backlog = backlog[b.size:]
            for m in miners:
                m.add_block(b, i)
    rewards = [0] * len(miners)
    blocks = [0] * len(miners)
    h = miners[0].head
    sz = 0
    while h is not None:
        rewards[h.miner] += REWARD + h.fees
        blocks[h.miner] += 1
        h = h.parent
        sz += 1
    return rewards, blocks, [m.created for m in miners]

for r in range(200):
    tests = []
    for s in strategies:
        tests.append(s)
        tests.append((s[0] - 2, s[1], s[2], s[3]))
        tests.append((s[0] + 2, s[1], s[2], s[3]))
        tests.append((s[0], s[1], s[2] - 1, s[3]))
        tests.append((s[0], s[1], s[2] + 1, s[3]))
        tests.append((s[0], s[1], s[2], s[3] - 2))
        tests.append((s[0], s[1], s[2], s[3] + 2))
    NUM_TESTS = 7
    print 'Starting simulation'
    results, blks, created = simulate(tests)
    for i, s in enumerate(strategies):
        base = results[i * NUM_TESTS]
        if results[i * NUM_TESTS + 1] < base < results[i * NUM_TESTS + 2]:
            print 'Increasing base accept size beneficial at %r' % s
            s[0] += 2
        if results[i * NUM_TESTS + 1] > base > results[i * NUM_TESTS + 2] and s[0] > 2:
            print 'Decreasing base accept size beneficial at %r' % s
            s[0] -= 2
        if results[i * NUM_TESTS + 3] < base < results[i * NUM_TESTS + 4]:
            print 'Increasing override depth beneficial at %r' % s
            s[2] += 1
        if results[i * NUM_TESTS + 3] > base > results[i * NUM_TESTS + 4] and s[2] > 1:
            print 'Decreasing override depth beneficial at %r' % s
            s[2] -= 1
        if results[i * NUM_TESTS + 5] < base < results[i * NUM_TESTS + 6]:
            print 'Increasing creation size beneficial at %r' % s
            s[3] += 2
        if results[i * NUM_TESTS + 5] > base > results[i * NUM_TESTS + 6] and s[3] > 2:
            print 'Decreasing creation size beneficial at %r' % s
            s[3] -= 2
    for s in strategies:
        print s
    print 'Chain quality (per miner):', [(b * 100 / c) if c else 0 for b, c in zip(blks, created)]
    print 'Chain quality (total, non-perturbed miners only):', sum(blks[::NUM_TESTS]) * 1.0 / sum(created[::NUM_TESTS])
    if r % 20 == 0:
        print 'Control round'
        results, blks, created = simulate(strategies)
        print 'Chain quality (per miner):', [(b * 100 / c) if c else 0 for b, c in zip(blks, created)]
        print 'Chain quality (total):', sum(blks) * 1.0 / sum(created)
        
    

# results = simulate(strategies)
# for s, r in zip(strategies, results):
#     print s[0], s[3], r
