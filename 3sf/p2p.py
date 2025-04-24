import random
import heapq
from typing import List, Dict, Union, Optional, Set
import copy
from consensus import (
    State, Vote, Block,
    process_block, get_latest_justified_block, get_fork_choice_head
)
from collections import defaultdict

SLOT_DURATION = 12  # time units

# A basic Staker node implementation
class Staker:
    def __init__(self, validator_id: int, network: 'P2PNetwork', genesis_block: Block, genesis_state: State):
        # This node's validator ID
        self.validator_id = validator_id
        # Hook to the p2p network
        self.network = network
        # {block hash: block} for all blocks that we know about
        self.chain: Dict[str, Block] = {}
        # {block hash: post state} for all blocks that we know about
        self.post_states: Dict[str, State] = {}
        self.inbox: List[Union[Block, Vote]] = []
        self.known_votes: List[Vote] = []
        self.dependencies: Dict[str, List[Block]] = {}
        self.new_votes: List[Vote] = []
        self.safe_target: Block = None
        self.chain["genesis"] = genesis_block
        self.post_states["genesis"] = genesis_state
        self.num_validators = genesis_state.config.num_validators
        self.network.register_staker(self)

    @property
    def head(self):
        justified_hash = get_latest_justified_block(self.post_states)
        return get_fork_choice_head(self.chain, justified_hash, self.known_votes)

    @property
    def latest_justified_block(self):
        return get_latest_justified_block(self.post_states)

    @property
    def latest_finalized_block(self):
        return self.post_states[self.head.hash].finalized_block

    def compute_safe_target(self):
        justified_hash = get_latest_justified_block(self.post_states)
        return get_fork_choice_head(
            self.chain,
            justified_hash,
            self.new_votes,
            min_score=self.num_validators * 2 // 3
        )

    def accept_new_votes(self):
        for vote in self.new_votes:
            if vote not in self.known_votes:
                self.known_votes.append(vote)
        self.new_votes = []

    def tick(self):
        self.process_received()
        time_in_slot = (self.network.time % SLOT_DURATION)
        if time_in_slot == 0:
            if self.get_current_slot() % self.num_validators == self.validator_id:
                self.accept_new_votes()
                self.propose_block()
        elif time_in_slot == SLOT_DURATION // 4:
            self.vote()
        elif time_in_slot == (SLOT_DURATION * 2) // 4:
            self.safe_target = self.compute_safe_target()
        elif time_in_slot == (SLOT_DURATION * 3) // 4:
            self.accept_new_votes()

    def get_current_slot(self):
        return self.network.time // SLOT_DURATION

    # Called when it's the staker's turn to propose a block
    def propose_block(self):
        new_slot = self.get_current_slot()
        block_hash = f"{self.validator_id}_slot_{new_slot}"

        head_state = self.post_states[self.head.hash]
        new_block, state = None, None
        votes_to_add = []
        # Keep attempt to add valid votes from the list of available votes
        while 1:
            new_block = Block(
                hash=block_hash,
                slot=new_slot,
                parent=self.head.hash,
                votes=votes_to_add
            )
            state = process_block(head_state, new_block)
            new_votes_to_add = [
                vote for vote in self.known_votes if
                vote.slot == state.latest_votes.get(vote.validator_id, 0) + 1
                and vote.source in state.justified_hashes
            ]

            if len(new_votes_to_add) == 0:
                break

            votes_to_add.extend(new_votes_to_add)
        new_block.state_root = state.compute_root()

        self.chain[new_block.hash] = new_block
        self.post_states[new_block.hash] = state

        self.network.submit(new_block, self.validator_id)

    # Called when it's the staker's turn to vote
    def vote(self):
        state = self.post_states[self.head.hash]
        head_block = self.head
        target_block = head_block
        safe_target = self.safe_target or self.chain["genesis"]
        for i in range(3):
            if target_block.slot > safe_target.slot:
                target_block = self.chain[target_block.parent]

        vote = Vote(
            validator_id=self.validator_id,
            slot=self.get_current_slot(),
            head=head_block.hash,
            head_slot=head_block.slot,
            target=target_block.hash,
            target_slot=target_block.slot,
            source=state.justified_block,
            source_slot=state.justified_slot
        )
        self.receive(vote)
        self.network.submit(vote, self.validator_id)

    # Called by the p2p network
    def receive(self, item: Union[Block, Vote]):
        self.inbox.append(item)

    # Called every time step
    def process_received(self):
        for item in self.inbox:
            self.process_item(item)
        self.inbox.clear()

    # Function to process an item that we received
    def process_item(self, item):
        if isinstance(item, Block):
            # If the block is already known, dump it
            if item.hash in self.chain:
                return
            parent_state = self.post_states.get(item.parent)
            if parent_state:
                state = process_block(copy.deepcopy(parent_state), item)
                self.chain[item.hash] = item
                self.post_states[item.hash] = state
                for vote in item.votes:
                    if vote not in self.known_votes:
                        self.known_votes.append(vote)
                # Once we have received a block, also process all of
                # its dependencies
                if item.hash in self.dependencies:
                    for item2 in self.dependencies[item.hash]:
                        self.process_item(item2)
                    del self.dependencies[item.hash]
            else:
                # If we have not yet seen the block's parent, ignore for now,
                # process later once we actually see the parent
                self.dependencies.setdefault(item.parent, []).append(item)
        elif isinstance(item, Vote):
            if item.head in self.chain:
                self.new_votes.append(item)
            else:
                self.dependencies.setdefault(item.head, []).append(item)

class P2PNetwork:
    def __init__(self):
        self.time = 0
        self.stakers: Dict[int, Staker] = {}
        self.queues: Dict[int, List[Tuple[int, Union[Block, Vote]]]] = defaultdict(list)

    def register_staker(self, staker: Staker):
        self.stakers[staker.validator_id] = staker

    def submit(self, item: Union[Block, Vote], sender_id: int):
        for recipient_id, staker in self.stakers.items():
            if recipient_id == sender_id:
                continue
            delay = int(SLOT_DURATION * 1.5 * random.random() ** 3)
            #delay = 1
            deliver_at = self.time + delay
            self.queues[recipient_id].append((deliver_at, item))

    def time_step(self):
        self.time += 1
        for validator_id, queue in self.queues.items():
            deliver_now = [item for (t, item) in queue if t <= self.time]
            self.queues[validator_id] = [(t, item) for (t, item) in queue if t > self.time]
            for item in deliver_now:
                self.stakers[validator_id].receive(item)
