from dataclasses import dataclass, field
from typing import Optional, List, Dict
import hashlib
import json
import copy

# Chain configuration
@dataclass
class Config:
    num_validators: int

# Blockchain state
@dataclass
class State:
    config: Config
    finalized_block: str
    finalized_slot: int
    justified_block: str
    justified_slot: int
    justified_hashes: set = field(default_factory=lambda: {"genesis"})
    justifications: Dict[tuple, List[bool]] = field(default_factory=dict)
    latest_votes: Dict[int, int] = field(default_factory=dict)

    # Stub for computing state root
    # (in real life replace with SSZ hashing)
    def compute_root(self):
        # Convert tuple keys to strings, sets to lists, and leave other values as is
        justifications_serializable = {
            f"{k[0]}->{k[1]}": v  # convert tuple (source, target) to "source->target"
            for k, v in self.justifications.items()
        }
    
        serializable_dict = {
            "finalized_block": self.finalized_block,
            "finalized_slot": self.finalized_slot,
            "justified_block": self.justified_block,
            "justified_slot": self.justified_slot,
            "justified_hashes": list(self.justified_hashes),
            "justifications": justifications_serializable
        }
    
        serialized = json.dumps(serializable_dict, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()

# A vote. In a live implementation this would also include a signature
@dataclass
class Vote:
    validator_id: int
    slot: int
    head: str
    head_slot: int
    target: str
    target_slot: int
    source: str
    source_slot: int

# A block
@dataclass
class Block:
    hash: str
    slot: int
    parent: Optional[str]
    votes: List[Vote] = field(default_factory=list)
    state_root: Optional[str] = None

# Given a state, output the new state after processing that block
def process_block(state: State, block: Block) -> State:
    state = copy.deepcopy(state)
    for vote in block.votes:
        key = (vote.source, vote.target)
        if vote.source not in state.justified_hashes:
            continue  # source must already be justified

        if vote.slot != state.latest_votes.get(vote.validator_id, 0) + 1:
            continue  # Must process votes in sequence

        state.latest_votes[vote.validator_id] = vote.slot

        if key not in state.justifications:
            state.justifications[key] = [False] * state.config.num_validators

        if not state.justifications[key][vote.validator_id]:
            state.justifications[key][vote.validator_id] = True

        count = sum(state.justifications[key])

        # Justification: if 2/3 voted for source=>target, and target.slot > current
        if count == (2 * state.config.num_validators) // 3:
            if vote.target not in state.justified_hashes:
                state.justified_hashes.add(vote.target)

            if vote.target_slot > state.justified_slot:
                state.justified_block = vote.target
                state.justified_slot = vote.target_slot

            # Finalization: if this is the latest justified, and this is an arrow k => k+1
            if  (
                vote.source_slot == vote.target_slot - 1 and
                vote.source_slot >= state.finalized_slot and
                vote.target_slot == state.justified_slot
                ):
                state.finalized_block = vote.source
                state.finalized_slot = vote.source_slot

    return state


# Get the highest-slot justified block that we know about
def get_latest_justified_block(post_states: Dict[str, State]) -> Block:
    latest = max(
        post_states.values(),
        key=lambda s: s.justified_slot
    )
    return latest.justified_block


# Use LMD GHOST to get the head, given a particular root (usually the
# latest known justified block)
def get_fork_choice_head(blocks: Dict[str, Block], root: str, votes: List[Vote]) -> str:
    children_map: Dict[str, List[str]] = {}
    for block in blocks.values():
        if block.parent:
            children_map.setdefault(block.parent, []).append(block.hash)

    latest_votes = []
    for v_id in set(vote.validator_id for vote in votes):
        latest_votes.append(max(
            [vote for vote in votes if vote.validator_id == v_id],
            key=lambda vote: vote.slot
        ))

    vote_weights: Dict[str, int] = {}

    for vote in latest_votes:
        block = blocks[vote.head]
        while block.slot > blocks[root].slot:
            vote_weights[block.hash] = vote_weights.get(block.hash, 0) + 1
            block = blocks[block.parent]

    current = root
    while True:
        children = children_map.get(current, [])
        if not children:
            return blocks[current]
        current = max(children, key=lambda x: vote_weights.get(x, 0))
