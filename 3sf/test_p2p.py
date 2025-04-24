from consensus import (
    State, Vote, Block, Config,
    get_latest_justified_block, get_fork_choice_head
)
from p2p import Staker, P2PNetwork
from typing import Optional, List, Dict
import random

SLOT_DURATION = 12
NUM_STAKERS = 10

import matplotlib.pyplot as plt
import networkx as nx


import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_view(fig, ax, staker: Staker, title="Staker's View"):
    G = nx.DiGraph()
    plt.clf()
    # First, build the graph structure
    children_map = defaultdict(list)
    for block in staker.chain.values():
        if block.parent:
            children_map[block.parent].append(block.hash)
        G.add_node(block.hash, slot=block.slot)

    for parent, children in children_map.items():
        for child in children:
            G.add_edge(parent, child)

    # DFS traversal to assign consistent x positions
    pos = {}
    x_counter = [0]
    max_validator_id = max(
        [staker.validator_id] +
        [vote.validator_id for vote in staker.known_votes]
    )

    def dfs(block_hash, depth=0):
        if block_hash not in children_map:
            x = x_counter[0]
            x_counter[0] += 1
            pos[block_hash] = (x, -staker.chain[block_hash].slot)
            return x
        child_xs = []
        for child in sorted(children_map[block_hash]):  # sort for determinism
            child_xs.append(dfs(child))
        x = sum(child_xs) / len(child_xs)
        pos[block_hash] = (x, -staker.chain[block_hash].slot)
        return x

    # Start DFS from genesis
    if "genesis" in staker.chain:
        dfs("genesis")

    # Color blocks
    justified_hash = get_latest_justified_block(staker.post_states)
    head_block = get_fork_choice_head(staker.chain, justified_hash, staker.known_votes)

    node_colors = []
    for node in G.nodes:
        if node == justified_hash:
            node_colors.append("blue")
        elif node == head_block.hash:
            node_colors.append("green")
        else:
            node_colors.append("black")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw votes
    latest_votes = {}
    for vote in staker.known_votes:
        latest_votes[vote.validator_id] = vote

    for vote in staker.known_votes:
        if vote.head not in pos:
            continue
        voter_node = f"v{vote.validator_id}"
        offset = (vote.validator_id - max_validator_id / 2) * 0.15
        pos[voter_node] = (pos[vote.head][0] + offset, pos[vote.head][1] - 0.5)

        G.add_node(voter_node, node_size=5)
        G.add_edge(voter_node, vote.head)
        G.add_edge(voter_node, vote.target)

        color = "orange" if latest_votes[vote.validator_id] == vote else "yellow"
        nx.draw_networkx_nodes(G, pos, nodelist=[voter_node], node_color=color, node_size=200)
        nx.draw_networkx_edges(G, pos, edgelist=[(voter_node, vote.head)], edge_color=color,
                               style="dashed", arrowsize=8)
        nx.draw_networkx_edges(G, pos, edgelist=[(voter_node, vote.target)], edge_color="grey",
                               style="dashed", arrowsize=8)
        nx.draw_networkx_labels(G, pos, labels={voter_node: f"v{vote.validator_id}"}, font_size=6)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__ == '__main__':
    SLOT_DURATION = 12
    NUM_STAKERS = 10


    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)

    # Create genesis block and state
    genesis_block = Block(hash="genesis", slot=0, parent=None)
    genesis_state = State(
        finalized_block="genesis",
        finalized_slot=0,
        justified_block="genesis",
        justified_slot=0,
        config=Config(num_validators=NUM_STAKERS)
    )
    genesis_block.state_root = genesis_state.compute_root()

    network = P2PNetwork()
    stakers = [Staker(i, network, genesis_block, genesis_state) for i in range(NUM_STAKERS)]

    # Initialize all stakers with genesis
    for staker in stakers:
        assert staker.head == genesis_block

    # Simulation loop
    for time in range(1, 1000):
        # Deliver messages
        network.time_step()

        # Run staker code
        for staker in stakers:
            staker.tick()

        # Periodic printout
        if time % SLOT_DURATION == 0:
            print(f"\n=== Time {time} ===")
            for staker in stakers:
                head = staker.head
                if head:
                    print(f"Staker {staker.validator_id}: Head={head.hash} | Justified={staker.latest_justified_block} | Finalized={staker.latest_finalized_block}")
        if time % 60 == 0:
            plot_view(fig, ax, stakers[0])
