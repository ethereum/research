from copy import deepcopy

from spec import (
    # constants
    DEPOSIT_CONTRACT_TREE_DEPTH,
    FAR_FUTURE_EPOCH,
    GENESIS_EPOCH,
    GENESIS_FORK_VERSION,
    MAX_DEPOSIT_AMOUNT,
    SLOTS_PER_EPOCH,
    ZERO_HASH,
    # SSZ
    BeaconBlock,
    Deposit,
    DepositData,
    DepositInput,
    Eth1Data,
    Fork,
    Validator,
    # functions
    int_to_bytes48,
    merkle_root,
    get_genesis_beacon_state,
    get_block_root,
    get_state_root,
    get_empty_block,
    advance_slot,
    process_block,
    state_transition,
    verify_merkle_branch,
)
from utils.merkle_normal import ( 
    verify_merkle_proof,
)
from utils.merkle_sparse import (
    calc_merkle_tree_from_leaves,
    get_merkle_proof,
    get_merkle_root,
)

from hashlib import sha256

def hash(x): return sha256(x).digest()


num_validators = 100
pubkeys = [int_to_bytes48(i) for i in range(100)]
all_deposit_data_leaves = list()


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


def get_empty_root():
    return get_merkle_root((ZERO_HASH,))


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


def create_mock_genesis_validator_deposits():
    withdrawal_credentials = b'\x22' * 32
    deposit_timestamp = 0
    proof_of_possession = b'\x33' * 96
    
    deposit_data_list = []
    for i in range(num_validators):
        deposit_data=DepositData(
            amount=MAX_DEPOSIT_AMOUNT,
            timestamp=deposit_timestamp,
            deposit_input=DepositInput(
                pubkey=pubkeys[i],
                withdrawal_credentials=withdrawal_credentials,
                proof_of_possession=proof_of_possession,
            ),
        )
        item = hash(deposit_data.serialize())
        all_deposit_data_leaves.append(item)
        tree = calc_merkle_tree_from_leaves(tuple(all_deposit_data_leaves))
        root = get_merkle_root((tuple(all_deposit_data_leaves)))
        proof = list(get_merkle_proof(tree, item_index=i))
        assert verify_merkle_branch(item, proof, DEPOSIT_CONTRACT_TREE_DEPTH, i, root)
        deposit_data_list.append(deposit_data)

    genesis_validator_deposits = []
    for i in range(num_validators):
        fork = Fork(
            previous_version=GENESIS_FORK_VERSION,
            current_version=GENESIS_FORK_VERSION,
            epoch=GENESIS_EPOCH,
        )
        genesis_validator_deposits.append(Deposit(
            proof=list(get_merkle_proof(tree, item_index=i)),
            index=i,
            deposit_data=deposit_data_list[i]
            
        )) 
    return genesis_validator_deposits, root


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
        print('get_block_root(test_state, slot)', get_block_root(test_state, slot))
        print('block.previous_block_root', block.previous_block_root)
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
    initial_deposits, deposit_root = create_mock_genesis_validator_deposits()
    genesis_state = get_genesis_beacon_state(
        initial_deposits,
        genesis_time=0,
        genesis_eth1_data=Eth1Data(
            deposit_root=deposit_root,
            block_hash=ZERO_HASH,
        ),
    )
    # add_validators_to_genesis(genesis_state, 100)
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