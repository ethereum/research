from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import hashlib
import json
import copy

ZERO_HASH = '0'*64

# Chain configuration
@dataclass
class Config:
    num_validators: int

# Blockchain state
@dataclass
class State:
    config: Config
    latest_justified_hash: str
    latest_justified_slot: int
    latest_finalized_hash: str
    latest_finalized_slot: int
    historical_block_hashes: List[str] = field(default_factory=list)
    justifications: Dict[str, List[bool]] = field(default_factory=dict)

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
    slot: int
    parent: Optional[str]
    votes: List[Vote] = field(default_factory=list)
    state_root: Optional[str] = None

# Stub for computing block hash, state root...
# (in real life replace with SSZ hashing)
def compute_hash(obj: object):
    serialized = json.dumps(asdict(obj), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

# We allow justification of slots either <= 5 or a perfect square after the
# latest finalized slot. This gives us a backoff technique and ensures finality
# keeps progressing even under high latency
def is_justifiable_slot(finalized_slot: int, candidate: int):
    assert candidate >= finalized_slot
    delta = candidate - finalized_slot
    return (
        delta <= 5
        or (delta ** 0.5) % 1 == 0            # any x^2
        or ((delta + 0.25) ** 0.5) % 1 == 0.5 # any x^2+x
    )

# Given a state, output the new state after processing that block
def process_block(state: State, block: Block) -> State:
    state = copy.deepcopy(state)
    # Track historical blocks in the state
    state.historical_block_hashes.append(block.parent)
    while len(state.historical_block_hashes) < block.slot:
        state.historical_block_hashes.append(None)
    # Process votes
    for vote in block.votes:
        # Ignore votes whose source is not the current latest justified hash,
        # or whose target is not in the history, or whose target is not a
        # valid justifiable slot
        if (
            vote.source != state.latest_justified_hash
            or vote.source != state.historical_block_hashes[vote.source_slot]
            or vote.target != state.historical_block_hashes[vote.target_slot]
            or vote.target_slot <= vote.source_slot
            or not is_justifiable_slot(state.latest_finalized_slot, vote.target_slot)
        ):
            continue

        # Track attempts to justify new hashes
        if vote.target not in state.justifications:
            state.justifications[vote.target] = [False] * state.config.num_validators

        if not state.justifications[vote.target][vote.validator_id]:
            state.justifications[vote.target][vote.validator_id] = True

        count = sum(state.justifications[vote.target])

        # If 2/3 voted for the same new valid hash to justify
        if count == (2 * state.config.num_validators) // 3:
            state.latest_justified_hash = vote.target
            state.latest_justified_slot = vote.target_slot
            state.justifications = {}

            # Finalization: if the target is the next valid justifiable
            # hash after the source
            if not any(
                is_justifiable_slot(state.latest_finalized_slot, slot)
                for slot in range(vote.source_slot + 1, vote.target_slot)
            ):
                state.latest_finalized_hash = vote.source
                state.latest_finalized_slot = vote.source_slot

    return state

# Get the highest-slot justified block that we know about
def get_latest_justified_hash(post_states: Dict[str, State]) -> str:
    latest = max(
        post_states.values(),
        key=lambda s: s.latest_justified_slot
    )
    return latest.latest_justified_hash


# Use LMD GHOST to get the head, given a particular root (usually the
# latest known justified block)
def get_fork_choice_head(blocks: Dict[str, Block],
                         root: str,
                         votes: List[Vote],
                         min_score: int = 0) -> str:
    if root == ZERO_HASH:
        # Start at genesis by default
        root = [h for h in blocks.keys() if blocks[h].slot == 1][0]

    # Identify latest votes
    latest_votes = {}
    for vote in sorted(votes, key=lambda vote: vote.slot):
        latest_votes[vote.validator_id] = vote

    # For each block, count the number of votes for that block. A vote
    # for any descendant of a block also counts as a vote for that block
    vote_weights: Dict[str, int] = {}

    for vote in latest_votes.values():
        if vote.head in blocks:
            block = blocks[vote.head]
            while block.slot > blocks[root].slot:
                h = compute_hash(block)
                vote_weights[h] = vote_weights.get(h, 0) + 1
                block = blocks[block.parent]

    # Identify the children of each block
    children_map: Dict[str, List[str]] = {}
    for block in blocks.values():
        h = compute_hash(block)
        if block.parent and vote_weights.get(h, 0) >= min_score:
            children_map.setdefault(block.parent, []).append(h)

    # Start at the root (generally, the latest justified hash) and
    # repeatedly choose the child with the higher number of latest votes
    current = root
    while True:
        children = children_map.get(current, [])
        if not children:
            return current
        current = max(children, key=lambda x: vote_weights.get(x, 0))
