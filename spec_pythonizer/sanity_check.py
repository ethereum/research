from copy import deepcopy

from spec import (
    # constants
    DEPOSIT_CONTRACT_TREE_DEPTH,
    FAR_FUTURE_EPOCH,
    GENESIS_EPOCH,
    GENESIS_FORK_VERSION,
    GENESIS_SLOT,
    MAX_DEPOSIT_AMOUNT,
    SLOTS_PER_EPOCH,
    ZERO_HASH,
    # SSZ
    BeaconBlock,
    BeaconBlockHeader,
    Deposit,
    DepositData,
    DepositInput,
    Eth1Data,
    Fork,
    ProposerSlashing,
    Validator,
    VoluntaryExit,
    # functions
    int_to_bytes48,
    merkle_root,
    get_active_validator_indices,
    get_current_epoch,
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
pubkeys = [int_to_bytes48(i) for i in range(10000)]
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
        assert get_block_root(test_state, slot) == block.previous_block_root


def test_empty_epoch_transition(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    for slot in range(state.slot, test_state.slot):
        assert get_block_root(test_state, slot) == block.previous_block_root


def test_proposer_slashing(state):
    test_state = deepcopy(state)
    current_epoch = get_current_epoch(test_state)
    validator_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[-1]
    slot = GENESIS_SLOT
    header_1 = BeaconBlockHeader(
        slot=slot,
        previous_block_root=b'\x00'*32,
        state_root=b'\x00'*32,
        block_body_root=b'\x00'*32,
        signature=b'\x00'*96
    )
    header_2 = deepcopy(header_1)
    header_2.previous_block_root = b'\x02'*32

    proposer_slashing = ProposerSlashing(
        proposer_index=validator_index,
        header_1=header_1,
        header_2=header_2,
    )

    #
    # Add to state via block transition
    #
    block = construct_empty_block_for_next_slot(test_state)
    block.body.proposer_slashings.append(proposer_slashing)
    state_transition(test_state, block)

    assert not state.validator_registry[validator_index].initiated_exit
    assert not state.validator_registry[validator_index].slashed

    slashed_validator = test_state.validator_registry[validator_index]
    assert not slashed_validator.initiated_exit
    assert slashed_validator.slashed
    assert slashed_validator.exit_epoch < FAR_FUTURE_EPOCH
    assert slashed_validator.withdrawable_epoch < FAR_FUTURE_EPOCH
    # lost whistleblower reward
    assert test_state.validator_balances[validator_index] < state.validator_balances[validator_index]


def test_deposit_in_block(state):
    test_state = deepcopy(state)
    test_deposit_data_leaves = deepcopy(all_deposit_data_leaves)
    withdrawal_credentials = b'\x42' * 32
    deposit_timestamp = 1
    proof_of_possession = b'\x44' * 96

    index = len(test_deposit_data_leaves)
    deposit_data = DepositData(
        amount=MAX_DEPOSIT_AMOUNT,
        timestamp=deposit_timestamp,
        deposit_input=DepositInput(
            pubkey=pubkeys[index],
            withdrawal_credentials=withdrawal_credentials,
            proof_of_possession=proof_of_possession,
        ),
    )
    item = hash(deposit_data.serialize())
    test_deposit_data_leaves.append(item)
    tree = calc_merkle_tree_from_leaves(tuple(test_deposit_data_leaves))
    root = get_merkle_root((tuple(test_deposit_data_leaves)))
    proof = list(get_merkle_proof(tree, item_index=index))
    assert verify_merkle_branch(item, proof, DEPOSIT_CONTRACT_TREE_DEPTH, index, root)

    deposit = Deposit(
        proof=list(proof),
        index=index,
        deposit_data=deposit_data,
    )

    test_state.latest_eth1_data.deposit_root = root
    block = construct_empty_block_for_next_slot(test_state)
    block.body.deposits.append(deposit)

    state_transition(test_state, block)
    assert len(test_state.validator_registry) == len(state.validator_registry) + 1
    assert len(test_state.validator_balances) == len(state.validator_balances) + 1
    assert test_state.validator_registry[index].pubkey == pubkeys[index]


def test_voluntary_exit(state):
    test_state = deepcopy(state)
    current_epoch = get_current_epoch(test_state)
    validator_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[-1]
    voluntary_exit = VoluntaryExit(
        epoch=current_epoch,
        validator_index=validator_index,
        signature=b'\x00'*96,
    )

    #
    # Add to state via block transition
    #
    block = construct_empty_block_for_next_slot(test_state)
    block.body.voluntary_exits.append(voluntary_exit)
    state_transition(test_state, block)

    assert not state.validator_registry[validator_index].initiated_exit
    assert test_state.validator_registry[validator_index].initiated_exit
    assert test_state.validator_registry[validator_index].exit_epoch == FAR_FUTURE_EPOCH

    #
    # Process within epoch transition
    #

    # artificially trigger registry update
    test_state.validator_registry_update_epoch -= 1

    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH
    state_transition(test_state, block)

    assert test_state.validator_registry[validator_index].exit_epoch < FAR_FUTURE_EPOCH


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
    test_proposer_slashing(genesis_state)
    test_deposit_in_block(genesis_state)
    test_voluntary_exit(genesis_state)
    print("done!")


if __name__ == "__main__":
    sanity_tests()