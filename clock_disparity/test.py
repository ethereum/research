from networksim import NetworkSimulator
from node import Node, NOTARIES, MainChainBlock, main_genesis
from distributions import normal_distribution

net = NetworkSimulator(latency=12)
notaries = [Node(i, net, ts=max(normal_distribution(50, 50)(), 0)) for i in range(NOTARIES)]
net.agents = notaries
net.generate_peers()
for i in range(4000):
    net.tick()
for n in notaries:
    print("Main chain head: %d" % n.blocks[n.main_chain[-1]].number)
    print("Total main chain blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, MainChainBlock)]) - 1))

import matplotlib.pyplot as plt
import networkx as nx
import random

G=nx.Graph()

#positions = {main_genesis.hash: 0, beacon_genesis.hash: 0}
#queue = [

for b in n.blocks.values():
    if b.number > 0:
        if isinstance(b, MainChainBlock):
            G.add_edge(b.hash, b.parent_hash, color='b')

cache = {main_genesis.hash: 0}

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
