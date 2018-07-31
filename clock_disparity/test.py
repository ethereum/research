from networksim import NetworkSimulator
from ghost_node import Node, NOTARIES, Block, genesis
from distributions import normal_distribution

net = NetworkSimulator(latency=22)
notaries = [Node(i, net, ts=max(normal_distribution(300, 300)(), 0) * 0.1, sleepy=i%4==0) for i in range(NOTARIES)]
net.agents = notaries
net.generate_peers()
for i in range(100000):
    net.tick()
for n in notaries:
    print("Local timestamp: %.1f, timequeue len %d" % (n.ts, len(n.timequeue)))
    print("Main chain head: %d" % n.blocks[n.main_chain[-1]].number)
    print("Total main chain blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, Block)]) - 1))
    print("Notarized main chain blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, Block) and n.is_notarized(b)]) - 1))

import matplotlib.pyplot as plt
import networkx as nx
import random

G=nx.Graph()

#positions = {genesis.hash: 0, beacon_genesis.hash: 0}
#queue = [

for b in n.blocks.values():
    for en in notaries:
        if isinstance(b, Block) and b.hash in en.processed and b.hash not in en.blocks:
            assert (not en.have_ancestry(b.hash)) or b.ts > en.ts
    if b.number > 0:
        if isinstance(b, Block):
            if n.is_notarized(b):
                G.add_edge(b.hash, b.parent_hash, color='b')
            else:
                G.add_edge(b.hash, b.parent_hash, color='#dddddd')


cache = {genesis.hash: 0}

def mkoffset(b):
    if b.hash not in cache:
        cache[b.hash] = cache[b.parent_hash] + random.randrange(35)
    return cache[b.hash]
        
pos={b.hash: (b.ts + mkoffset(b), b.ts) for b in n.blocks.values()}
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw_networkx_nodes(G,pos,node_size=10,node_shape='o',node_color='0.75')

nx.draw_networkx_edges(G,pos,
                width=2,edge_color=colors)

plt.axis('off')
# plt.savefig("degree.png", bbox_inches="tight")
plt.show() 
