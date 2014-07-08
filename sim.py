# Time between block number increases (NOT time between PoW solutions)
BLOCKTIME = 50
# Time for a block to traverse the network
TRANSIT_TIME = 50
# Number of required uncles
UNCLES = 4
# Uncle block reward (normal block reward = 1)
UNCLE_REWARD_COEFF = 0.875
# Block length to test
BLOCKS = 5000

import random
import copy


class Miner():
    def __init__(self, p):
        self.hashpower = p
        self.id = random.randrange(1000000000)
        # Set up a few genesis blocks (since the algo is grandpa-dependent,
        # we need two genesis blocks plus some genesis uncles)
        self.blocks = {
            0: {"parent": -1, "uncles": [], "miner": -1, "height": 0,
                "score": 0, "id": 0, "children": {1: 1}},
            1: {"parent": 0, "uncles": [], "miner": -1, "height": 1,
                "score": 0, "id": 1, "children": {}}
        }
        # for i in range(2, UNCLES + 2):
        #     self.blocks[i] = {"parent": 0, "uncles": [], "miner": -1,
        #                       "height": 1, "id": i, "children": {}}
        # ID of parent of "latest block"
        # We care about the parent of the latest block because that way
        # we can keep track of uncles.
        self.head = 0

    # Hear about a block
    def recv(self, block):
        # Add the block to the set if it's valid
        addme = True
        if block["id"] in self.blocks:
            addme = False
        if block["parent"] not in self.blocks:
            addme = False
        for u in block["uncles"]:
            if self.blocks[u]["id"] not in self.blocks:
                addme = False
        p = self.blocks[block["parent"]]
        if addme:
            self.blocks[block["id"]] = copy.deepcopy(block)
            # Check if the new block's parent deserves to be the new head
            if block["id"] not in p["children"]:
                p["children"][block["id"]] = block["id"]
            if len(p["children"]) >= 1 + UNCLES:
                if block["score"] > self.blocks[self.head]["score"]:
                    self.head = block["parent"]

    # Mine a block
    def mine(self):
        h = self.blocks[self.head]
        b = sorted(list(h["children"]), key=lambda x: self.blocks[x]["id"])
        p = self.blocks[b[0]]
        block = {"parent": b[0], "uncles": b[1:], "miner": self.id,
                 "height": h["height"] + 2, "score": p["score"] + len(b),
                 "id": random.randrange(1000000000), "children": {}}
        self.recv(block)
        return block

percentages = [1]*25 + [5, 5, 5, 5, 5, 10, 15, 25]
miners = []
for p in percentages:
    miners.append(Miner(p))

miner_dict = {}
for m in miners:
    miner_dict[m.id] = m

listen_queue = []

for t in range(BLOCKS * BLOCKTIME):
    if t % 5000 == 0:
        print t
    for m in miners:
        R = random.randrange(BLOCKTIME * sum(percentages) / (UNCLES + 1))
        if R < m.hashpower:
            b = m.mine()
            listen_queue.append([t + TRANSIT_TIME, b])
    while len(listen_queue) and listen_queue[0][0] <= t:
        t, b = listen_queue.pop(0)
        for m in miners:
            m.recv(b)

h = miners[0].blocks[miners[0].head]
profit = {}
print "### PRINTING BLOCKCHAIN ###"

while h["id"] > 1:
    print h["miner"], h["height"], h["score"]
    profit[h["miner"]] = profit.get(h["miner"], 0) + 1
    for u in h["uncles"]:
        u2 = miners[0].blocks[u]
        profit[u2["miner"]] = profit.get(u2["miner"], 0) + UNCLE_REWARD_COEFF
    h = miners[0].blocks[h["parent"]]

print "### PRINTING PROFITS ###"

for p in profit:
    print miner_dict[p].hashpower, profit[p]

print "### PRINTING GROUPINGS ###"

groupings = {}
counts = {}
for p in profit:
    h = miner_dict[p].hashpower
    counts[h] = counts.get(h, 0) + 1
    groupings[h] = groupings.get(h, 0) + profit[p]

for c in counts:
    print c, groupings[c] / counts[c] / (groupings[1] / counts[1])
