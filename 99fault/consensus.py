# Implementation of 99% fault tolerant consensus as decribed by Leslie Lamport
# on page 391 of https://people.eecs.berkeley.edu/~luca/cs174/byzantine.pdf

import random
from network import Network

TIMEOUT = 10
LATENCY = 5
# A node in the network (not including the "commander")
class Node():

    def __init__(self, network, id, honest=True):
        self.seen = {}
        self.id = id
        self.network = network
        self.network.add_node(self)
        self.honest = honest

    # Upon receiving a message...
    def on_receive(self, msg):
        if self.id == 0 and len(msg) == 1:
            print('Seen', msg[0], self.network.time, 'already seen', sorted(self.seen.keys()))
        # Only proceed if v not in V_i...
        # Byzantine nodes ignore this rule 5% of the time
        if (msg[0] in self.seen) and (self.honest or random.random() < 0.95):
            return
        # Timeout logic (see page 399)
        # Byzantine nodes ignore this rule
        if (len(msg) * TIMEOUT < self.network.time) and self.honest:
            return
        self.seen[msg[0]] = True
        new_msg = msg + [self.id]
        # Broadcast v:0:j1...jk:i
        # Byzantine nodes delay broadcast to split honest receiving nodes by timeout
        self.network.broadcast(new_msg, at=self.network.time if self.honest else self.network.time + int(TIMEOUT / 1.5))

    def choice(self):
        return max(self.seen.keys())

def test():
    n = Network(LATENCY * 2)
    nodes = [Node(n, i, i%4==0) for i in range(20)]
    for i in range(30):
        for _ in range(2):
            z = random.randrange(20)
            n.send_to([1000 + i], z, at=5+i*2)
    for i in range(21 * LATENCY):
        n.tick()
        if i % 10 == 0:
            print("Value sets", [sorted(node.seen.keys()) for node in nodes])
    countz = {}
    maxval = ""
    for node in nodes:
        if node.honest:
            k = str(sorted(node.seen.keys()))
            countz[k] = countz.get(k, 0) + 1
            if countz[k] > countz.get(maxval, 0):
                maxval = k
    print("Most popular: %s" % maxval, "with %d agreeing" % countz[maxval])

if __name__ == '__main__':
    test()
