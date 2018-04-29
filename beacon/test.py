from networksim import NetworkSimulator
from beacon_chain_node import Node, NOTARIES, BeaconBlock, MainChainBlock, main_genesis, beacon_genesis

net = NetworkSimulator(latency=15)
notaries = [Node(i, net, sleepy=i % 5 == 9) for i in range(NOTARIES)]
net.agents = notaries
net.generate_peers()
for i in range(2000):
    net.tick()
for n in notaries:
    print("Beacon head: %d" % n.blocks[n.beacon_head].number)
    print("Main chain head: %d" % n.blocks[n.main_chain[-1]].number)
    print("Total beacon blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, BeaconBlock)]) - 1))
    print("Total beacon blocks received and signed: %d" % (len([b for b in n.blocks.keys() if b in n.sigs and len(n.sigs[b]) >= n.blocks[b].notary_req]) - 1))
    print("Total main chain blocks received: %d" % (len([b for b in n.blocks.values() if isinstance(b, MainChainBlock)]) - 1))

import matplotlib.pyplot as plt
import networkx as nx
import random

G=nx.Graph()

#positions = {main_genesis.hash: 0, beacon_genesis.hash: 0}
#queue = [

for b in n.blocks.values():
    if b.number > 0:
        if isinstance(b, BeaconBlock):
            G.add_edge(b.hash, b.main_chain_ref, color='g')
            G.add_edge(b.hash, b.parent_hash, color='y')
        else:
            G.add_edge(b.hash, b.parent_hash, color='b')
        
#G.add_edge('a','b',weight=1)
#G.add_edge('a','c',weight=1)
#G.add_edge('a','d',weight=1)
#G.add_edge('a','e',weight=1)
#G.add_edge('a','f',weight=1)
#G.add_edge('a','g',weight=1)

# pos=nx.spring_layout(G)
ypos={main_genesis.hash: 0, beacon_genesis.hash: 0}
queue = n.children[main_genesis.hash] + n.children[beacon_genesis.hash]
while len(queue):
    first = queue.pop(0)
    if isinstance(n.blocks[first], MainChainBlock):
        if n.blocks[first].parent_hash not in ypos:
            queue.append(first)
            continue
        ypos[first] = ypos[n.blocks[first].parent_hash] + 10
    elif isinstance(n.blocks[first], BeaconBlock):
        if n.blocks[first].parent_hash not in ypos or n.blocks[first].main_chain_ref not in ypos:
            queue.append(first)
            continue
        ypos[first] = max(ypos[n.blocks[first].parent_hash] + 1, ypos[n.blocks[first].main_chain_ref] + 1)
    if first in n.children:
        queue.extend(n.children[first])
pos={b.hash: (b.ts + random.randrange(5) + (5 if isinstance(b, MainChainBlock) else 0), b.ts) for b in n.blocks.values()}
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw_networkx_nodes(G,pos,node_size=10,node_shape='o',node_color='0.75')

nx.draw_networkx_edges(G,pos,
                width=2,edge_color=colors)

plt.axis('off')
# plt.savefig("degree.png", bbox_inches="tight")
plt.show() 
