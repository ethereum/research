from copy import deepcopy

from spec import (
    FAR_FUTURE_EPOCH,
    GENESIS_EPOCH,
    MAX_DEPOSIT_AMOUNT,
    SLOTS_PER_EPOCH,
    ZERO_HASH,
    BeaconBlock,
    DepositData,
    DepositInput,
    Eth1Data,
    Validator,
    int_to_bytes48,
    merkle_root,
    get_genesis_beacon_state,
    get_block_root,
    get_state_root,
    get_empty_block,
    advance_slot,
    process_block,
    state_transition,
)


def get_sample_genesis_validator(index):
    return Validator(
        pubkey=int_to_bytes48(index),
        withdrawal_credentials=ZERO_HASH,
        activation_epoch=GENESIS_EPOCH,
        exit_epoch=FAR_FUTURE_EPOCH,
        withdrawable_epoch=FAR_FUTURE_EPOCH,
        initiated_exit=False,
        slashed=False,
    )


def add_validators_to_genesis(state, num_validators):
    # currently bypassing normal deposit route
    # TODO: get merkle root working and use normal genesis_deposits
    state.validator_registry = [
        get_sample_genesis_validator(i)
        for i in range(num_validators)
    ]
    state.validator_balances = [
        int(MAX_DEPOSIT_AMOUNT) for i in range(num_validators)
    ]


def construct_empty_block_for_next_slot(state):
    empty_block = get_empty_block()
    empty_block.slot = state.slot + 1
    previous_block_header = deepcopy(state.latest_block_header)
    if previous_block_header.state_root == ZERO_HASH:
        previous_block_header.state_root = state.hash_tree_root()
    empty_block.previous_block_root = previous_block_header.hash_tree_root()
    return empty_block


def test_slot_transition(state):
    test_state = deepcopy(state)
    advance_slot(test_state)
    assert test_state.slot == state.slot + 1
    assert get_state_root(test_state, state.slot) == state.hash_tree_root()
    return test_state


def test_empty_block_transition(state):
    test_state = deepcopy(state)

    block = construct_empty_block_for_next_slot(state)
    advance_slot(test_state)
    process_block(test_state, block)

    assert len(test_state.eth1_data_votes) == len(state.eth1_data_votes) + 1
    assert get_block_root(test_state, state.slot) == block.previous_block_root


def test_skipped_slots(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += 3

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    for slot in range(state.slot, test_state.slot):
        assert get_block_root(test_state, slot) == block.previous_block_root


def test_empty_epoch_transition(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    for slot in range(state.slot, test_state.slot):
        assert get_block_root(test_state, slot) == block.previous_block_root


def sanity_tests():
    print("Buidling state with 100 validators...")
    genesis_state = get_genesis_beacon_state(
        [],
        0,
        Eth1Data(
            deposit_root="\x00"*32,
            block_hash="\x00"*32
        ),
    )
    add_validators_to_genesis(genesis_state, 100)
    print("done!")
    print()

    print("Running some sanity check tests...")
    test_slot_transition(genesis_state)
    test_empty_block_transition(genesis_state)
    test_skipped_slots(genesis_state)
    test_empty_epoch_transition(genesis_state)
    print("done!")


if __name__ == "__main__":
    sanity_tests()