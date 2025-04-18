from consensus import (
    State, Vote, Block, Config,
    get_latest_justified_block, get_fork_choice_head, process_block
)
from typing import Optional, List, Dict

if __name__ == '__main__':
    # Genesis block and state
    genesis = Block(hash="genesis", slot=0, parent=None)
    state = State(
        finalized_block="genesis",
        finalized_slot=0,
        justified_block="genesis",
        justified_slot=0,
        config=Config(num_validators=100)
    )
    genesis.state_root = state.compute_root()
    chain: Dict[str, Block] = {"genesis": genesis}
    post_states: Dict[str, State] = {"genesis": state}
    all_votes: List[Vote] = []
    parent = genesis

    for step in range(1, 7):
        block_hash = f"block_{step}"
        new_block = Block(
            hash=block_hash,
            slot=step,
            parent=parent.hash
        )

        # Get latest justified and fork-choice head
        source_block = chain[get_latest_justified_block(post_states)]
        head = get_fork_choice_head(chain, source_block.hash, all_votes)
        head_hash = head.hash
        target_block = head
        if target_block.slot == new_block.slot - 1 and target_block.slot > 0:
            target_block = chain[target_block.parent]
        print('v', head.slot, target_block.slot, source_block.slot)

        for validator_id in range(80):
            vote = Vote(
                validator_id=validator_id,
                slot=step,
                head=head_hash,
                head_slot=chain[head_hash].slot,
                target=target_block.hash,
                target_slot=target_block.slot,
                source=source_block.hash,
                source_slot=source_block.slot
            )
            new_block.votes.append(vote)
            all_votes.append(vote)

        new_block.state_root = state.compute_root()
        chain[block_hash] = new_block
        state = process_block(state, new_block)
        post_states[block_hash] = state
        parent = new_block

        print(f"Time slot {step}:")
        print(f"  Head: {new_block.hash} (slot {new_block.slot})")
        print(f"  Justified: {state.justified_block} (slot {state.justified_slot})")
        print(f"  Finalized: {state.finalized_block} (slot {state.finalized_slot})")
        print()
