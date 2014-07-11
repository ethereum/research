# Time between successful PoW solutions
POW_SOLUTION_TIME = 10
# Time for a block to traverse the network
TRANSIT_TIME = 50
# Number of required uncles
UNCLES = 4
# Uncle block reward (normal block reward = 1)
UNCLE_REWARD_COEFF = 0.875
# Reward for including uncles
NEPHEW_REWARD_COEFF = 0.01
# Rounds to test
ROUNDS = 80000

import random
import copy


class Miner():
    def __init__(self, p):
        self.hashpower = p
        self.id = random.randrange(10000000)
        # Set up a few genesis blocks (since the algo is grandpa-dependent,
        # we need two genesis blocks plus some genesis uncles)
        self.blocks = {
            0: {"parent": -1, "uncles": [], "miner": -1, "height": 0,
                "score": 0, "id": 0, "children": {1: 1}},
            1: {"parent": 0, "uncles": [], "miner": -1, "height": 1,
                "score": 0, "id": 1, "children": {}}
        }
        # ID of "latest block"
        self.head = 1

    # Hear about a block
    def recv(self, block):
        # Add the block to the set if it's valid
        addme = True
        if block["id"] in self.blocks:
            addme = False
        if block["parent"] not in self.blocks:
            addme = False
        for u in block["uncles"]:
            if u not in self.blocks:
                addme = False
        p = self.blocks[block["parent"]]
        if addme:
            self.blocks[block["id"]] = copy.deepcopy(block)
            # Each parent keeps track of its children, to help
            # facilitate the rule that a block must have N+ siblings
            # to be valid
            if block["id"] not in p["children"]:
                p["children"][block["id"]] = block["id"]
            # Check if the new block deserves to be the new head
            if len(p["children"]) >= 1 + UNCLES:
                for c in p["children"]:
                    newblock = self.blocks[c]
                    if newblock["score"] > self.blocks[self.head]["score"]:
                        self.head = newblock["id"]

    # Mine a block
    def mine(self):
        h = self.blocks[self.blocks[self.head]["parent"]]
        b = sorted(list(h["children"]), key=lambda x: -self.blocks[x]["score"])
        p = self.blocks[b[0]]
        block = {"parent": b[0], "uncles": b[1:], "miner": self.id,
                 "height": h["height"] + 2, "score": p["score"] + len(b),
                 "id": random.randrange(1000000000000), "children": {}}
        self.recv(block)
        return block


def cousin_degree(miner, b1, b2):
    while miner.blocks[b1]["height"] > miner.blocks[b2]["height"]:
        b1 = miner.blocks[b1]["parent"]
    while miner.blocks[b2]["height"] > miner.blocks[b1]["height"]:
        b2 = miner.blocks[b2]["parent"]
    t = 0
    while b1 != b2:
        b1 = miner.blocks[b1]["parent"]
        b2 = miner.blocks[b2]["parent"]
        t += 1
    return t

percentages = [1]*25 + [5, 5, 5, 5, 5, 10, 15, 25]
miners = []
for p in percentages:
    miners.append(Miner(p))

miner_dict = {}
for m in miners:
    miner_dict[m.id] = m

listen_queue = []

for t in range(ROUNDS):
    if t % 5000 == 0:
        print t
    for m in miners:
        R = random.randrange(POW_SOLUTION_TIME * sum(percentages))
        if R < m.hashpower and t < ROUNDS - TRANSIT_TIME * 3:
            b = m.mine()
            listen_queue.append([t + TRANSIT_TIME, b])
    while len(listen_queue) and listen_queue[0][0] <= t:
        t, b = listen_queue.pop(0)
        for m in miners:
            m.recv(b)

h = miners[0].blocks[miners[0].head]
profit = {}
total_blocks_in_chain = 0
length_of_chain = 0
ZORO = {}
print "### PRINTING BLOCKCHAIN ###"

while h["id"] > 1:
    print h["miner"], h["height"], h["score"]
    total_blocks_in_chain += 1 + len(h["uncles"])
    ZORO[h["id"]] = True
    length_of_chain += 1
    profit[h["miner"]] = profit.get(h["miner"], 0) + \
        1 + NEPHEW_REWARD_COEFF * len(h["uncles"])
    for u in h["uncles"]:
        ZORO[u] = True
        u2 = miners[0].blocks[u]
        profit[u2["miner"]] = profit.get(u2["miner"], 0) + UNCLE_REWARD_COEFF
    h = miners[0].blocks[h["parent"]]

print "### PRINTING HEADS ###"

for m in miners:
    print m.head


print "### PRINTING PROFITS ###"

for p in profit:
    print miner_dict[p].hashpower, profit[p]

print "### PRINTING RESULTS ###"

groupings = {}
counts = {}
for p in profit:
    h = miner_dict[p].hashpower
    counts[h] = counts.get(h, 0) + 1
    groupings[h] = groupings.get(h, 0) + profit[p]

for c in counts:
    print c, groupings[c] / counts[c] / (groupings[1] / counts[1])

print " "
print "Total blocks produced: ", len(miners[0].blocks) - 2
print "Total blocks in chain: ", total_blocks_in_chain
print "Efficiency: ", total_blocks_in_chain * 1.0 / (len(miners[0].blocks) - 2)
print "Average uncles: ", total_blocks_in_chain * 1.0 / length_of_chain
print "Length of chain: ", length_of_chain
print "Block time: ", ROUNDS * 1.0 / length_of_chain
