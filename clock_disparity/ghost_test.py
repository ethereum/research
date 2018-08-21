from networksim import NetworkSimulator
from ghost_node import Node, NOTARIES, Block, Sig, genesis, SLOT_SIZE
from distributions import normal_distribution

net = NetworkSimulator(latency=45)
notaries = [Node(i, net, ts=max(normal_distribution(5, 5)(), 0) * 0.1, sleepy=False) for i in range(NOTARIES)]
net.agents = notaries
net.generate_peers()
for i in range(12000):
    net.tick()
for n in notaries:
    print("Local timestamp: %.1f, timequeue len %d" % (n.ts, len(n.timequeue)))
    print("Main chain head: %d" % n.blocks[n.main_chain[-1]].height)
    print("Total main chain blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, Block)]) - 1))

import matplotlib.pyplot as plt
import networkx as nx
import random

G=nx.Graph()

#positions = {genesis.hash: 0, beacon_genesis.hash: 0}
#queue = [

for b in n.blocks.values():
    if b.height > 0:
        if isinstance(b, Block):
            G.add_edge(b.hash, b.parent_hash, color='b')
for s in n.sigs.values():
    G.add_edge(s.hash, s.targets[0], color='0.75')


cache = {genesis.hash: 0}

def mkoffset(b):
    if b.hash not in cache:
        cache[b.hash] = cache[b.parent_hash] + random.randrange(35)
    return cache[b.hash]
        
pos={b'\x00'*32: (0, 0)}
for b in sorted(n.blocks.values(), key=lambda b: b.height):
    x,y = pos[b.parent_hash]
    pos[b.hash] = (x + (random.randrange(5) if b.hash in n.main_chain else -random.randrange(5)), y+10)
for s in n.sigs.values():
    parent = n.blocks[s.targets[0]]    
    x,y = pos[parent.hash]
    pos[s.hash] = (x - 2 + random.randrange(5),
                   y + 5)

finalized = {k:v for k,v in pos.items() if k in n.finalized}
justified = {k:v for k,v in pos.items() if k in n.justified and k not in n.finalized}
unjustified = {k:v for k,v in pos.items() if k not in n.justified and k in n.blocks}
sigs = {k:v for k,v in pos.items() if k not in n.blocks}

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]

nx.draw_networkx_nodes(G, pos, nodelist=sigs.keys(), node_size=5, node_shape='o',node_color='0.75')
nx.draw_networkx_nodes(G, pos, nodelist=unjustified.keys(), node_size=10, node_shape='o',node_color='0.75')
nx.draw_networkx_nodes(G, pos, nodelist=justified.keys(), node_size=16, node_shape='o',node_color='y')
nx.draw_networkx_nodes(G, pos, nodelist=finalized.keys(), node_size=25, node_shape='o',node_color='g')
# nx.draw_networkx_labels(G, pos, {h: n.scores.get(h, 0) for h in n.blocks.keys()}, font_size=5)

blockedges = [(u,v) for (u,v) in edges if G[u][v]['color'] == 'b']
otheredges = [(u,v) for (u,v) in edges if G[u][v]['color'] == '0.75']
nx.draw_networkx_edges(G, pos, edgelist=otheredges, width=1, edge_color='0.75')
nx.draw_networkx_edges(G, pos, edgelist=blockedges, width=2, edge_color='b')

print('Scores:', [n.scores.get(c, 0) for c in n.main_chain])

plt.axis('off')
# plt.savefig("degree.png", bbox_inches="tight")
plt.show() 
