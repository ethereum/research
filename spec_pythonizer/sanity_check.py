from copy import deepcopy
import time
import sys
import json
from jsonize import jsonize

import spec
from spec import (
    # constants
    BLS_WITHDRAWAL_PREFIX_BYTE,
    DEPOSIT_CONTRACT_TREE_DEPTH,
    EJECTION_BALANCE,
    FAR_FUTURE_EPOCH,
    GENESIS_EPOCH,
    GENESIS_SLOT,
    MAX_DEPOSIT_AMOUNT,
    MIN_ATTESTATION_INCLUSION_DELAY,
    PERSISTENT_COMMITTEE_PERIOD,
    SLOTS_PER_EPOCH,
    SLOTS_PER_HISTORICAL_ROOT,
    ZERO_HASH,
    # SSZ
    Bytes32,
    List,
    Epoch,
    Attestation,
    AttestationData,
    BeaconBlockHeader,
    Deposit,
    DepositData,
    DepositInput,
    Eth1Data,
    Transfer,
    ProposerSlashing,
    Validator,
    ValidatorIndex,
    VoluntaryExit,
    # functions
    int_to_bytes48,
    get_active_validator_indices,
    get_current_epoch,
    get_crosslink_committees_at_slot,
    get_epoch_start_slot,
    get_genesis_beacon_state,
    get_block_root,
    get_state_root,
    get_empty_block,
    advance_slot,
    state_transition,
    cache_state,
    verify_merkle_branch,
    hash_tree_root,
    hash
)
from utils.merkle_minimal import (
    calc_merkle_tree_from_leaves,
    get_merkle_proof,
    get_merkle_root,
)

from hashlib import sha256

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r  %2.2f ms' % \
              (method.__name__, (te - ts) * 1000))

        return result

    return timed

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


def create_mock_genesis_validator_deposits(num_validators=100):
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


def create_genesis_state(num_validators=100, genesis_time=0):
    initial_deposits, deposit_root = create_mock_genesis_validator_deposits(num_validators)
    return get_genesis_beacon_state(
        initial_deposits,
        genesis_time=genesis_time,
        genesis_eth1_data=Eth1Data(
            deposit_root=deposit_root,
            block_hash=ZERO_HASH,
        ),
    )


def construct_empty_block_for_next_slot(state):
    empty_block = get_empty_block()
    empty_block.slot = state.slot + 1
    previous_block_header = deepcopy(state.latest_block_header)
    if previous_block_header.state_root == ZERO_HASH:
        previous_block_header.state_root = state.hash_tree_root()
    empty_block.previous_block_root = previous_block_header.hash_tree_root()
    return empty_block


def build_attestation_data(state, slot, shard):
    assert state.slot >= slot

    if state.slot == slot:
        block_root = construct_empty_block_for_next_slot(state).previous_block_root
    else:
        block_root = get_block_root(state, slot)

    epoch_start_slot = get_epoch_start_slot(get_current_epoch(state))
    if epoch_start_slot == slot:
        epoch_boundary_root = block_root
    else:
        get_block_root(state, epoch_start_slot)

    justified_epoch_slot = get_epoch_start_slot(state.justified_epoch)
    if justified_epoch_slot == slot:
        justified_block_root = block_root
    else:
        justified_block_root = get_block_root(state, justified_epoch_slot)

    return AttestationData(
        slot=slot,
        shard=shard,
        beacon_block_root=block_root,
        epoch_boundary_root=epoch_boundary_root,
        crosslink_data_root=ZERO_HASH,
        latest_crosslink=deepcopy(state.latest_crosslinks[shard]),
        justified_epoch=state.justified_epoch,
        justified_block_root=justified_block_root,
    )


@timeit
def test_slot_transition(state):
    test_state = deepcopy(state)
    cache_state(test_state)
    advance_slot(test_state)
    assert test_state.slot == state.slot + 1
    assert get_state_root(test_state, state.slot) == state.hash_tree_root()
    return test_state


@timeit
def test_empty_block_transition(state):
    test_state = deepcopy(state)

    block = construct_empty_block_for_next_slot(test_state)
    state_transition(test_state, block)

    assert len(test_state.eth1_data_votes) == len(state.eth1_data_votes) + 1
    assert get_block_root(test_state, state.slot) == block.previous_block_root

    return state, [block], test_state


@timeit
def test_skipped_slots(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += 3

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    for slot in range(state.slot, test_state.slot):
        assert get_block_root(test_state, slot) == block.previous_block_root

    return state, [block], test_state


@timeit
def test_empty_epoch_transition(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    for slot in range(state.slot, test_state.slot):
        assert get_block_root(test_state, slot) == block.previous_block_root

    return state, [block], test_state


@timeit
def test_empty_epoch_transition_not_finalizing(state):
    test_state = deepcopy(state)
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH * 5

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    assert test_state.finalized_epoch < get_current_epoch(test_state) - 4

    return state, [block], test_state


@timeit
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

    return state, [block], test_state


@timeit
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

    return state, [block], test_state


@timeit
def test_attestation(state):
    test_state = deepcopy(state)
    current_epoch = get_current_epoch(test_state)
    slot = state.slot
    shard = state.current_shuffling_start_shard
    attestation_data = build_attestation_data(state, slot, shard)

    crosslink_committees = get_crosslink_committees_at_slot(state, slot)
    crosslink_committee = [committee for committee, _shard in crosslink_committees if _shard == attestation_data.shard][0]

    committee_size = len(crosslink_committee)
    bitfield_length = (committee_size + 7) // 8
    aggregation_bitfield = b'\x01' + b'\x00' * (bitfield_length - 1)
    custody_bitfield = b'\x00' * bitfield_length
    attestation = Attestation(
        aggregation_bitfield=aggregation_bitfield,
        data=attestation_data,
        custody_bitfield=custody_bitfield,
        aggregate_signature=b'\x42'*96,
    )

    #
    # Add to state via block transition
    #
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += MIN_ATTESTATION_INCLUSION_DELAY
    block.body.attestations.append(attestation)
    state_transition(test_state, block)

    assert len(test_state.current_epoch_attestations) == len(state.current_epoch_attestations) + 1

    #
    # Epoch transition should move to previous_epoch_attestations
    #
    pre_current_epoch_attestations = deepcopy(test_state.current_epoch_attestations)

    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH
    state_transition(test_state, block)

    assert len(test_state.current_epoch_attestations) == 0
    assert test_state.previous_epoch_attestations == pre_current_epoch_attestations

    return state, [block], test_state


@timeit
def test_voluntary_exit(state):
    test_state = deepcopy(state)
    current_epoch = get_current_epoch(test_state)
    validator_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[-1]
    voluntary_exit = VoluntaryExit(
        epoch=current_epoch,
        validator_index=validator_index,
        signature=b'\x00'*96,
    )

    # move state forward PERSISTENT_COMMITTEE_PERIOD epochs to allow for exit
    test_state.slot += PERSISTENT_COMMITTEE_PERIOD * SLOTS_PER_EPOCH

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

    return state, [block], test_state


@timeit
def test_transfer(state):
    test_state = deepcopy(state)
    current_epoch = get_current_epoch(test_state)
    sender_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[-1]
    recipient_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[0]
    pubkey = b'\x00' * 48
    amount = test_state.validator_balances[sender_index]
    pre_transfer_recipient_balance = test_state.validator_balances[recipient_index]
    transfer = Transfer(
        sender=sender_index,
        recipient=recipient_index,
        amount=amount,
        fee=0,
        slot=test_state.slot + 1,
        pubkey=pubkey,
        signature=b'\x42'*96,
    )

    # ensure withdrawal_credentials reproducable
    test_state.validator_registry[sender_index].withdrawal_credentials = (
        BLS_WITHDRAWAL_PREFIX_BYTE + hash(pubkey)[1:]
    )
    # un-activate so validator can transfer
    test_state.validator_registry[sender_index].activation_epoch = FAR_FUTURE_EPOCH

    #
    # Add to state via block transition
    #
    block = construct_empty_block_for_next_slot(test_state)
    block.body.transfers.append(transfer)
    state_transition(test_state, block)

    sender_balance = test_state.validator_balances[sender_index]
    recipient_balance = test_state.validator_balances[recipient_index]
    assert sender_balance == 0
    assert recipient_balance == pre_transfer_recipient_balance + amount

    return state, [block], test_state


@timeit
def test_ejection(state):
    test_state = deepcopy(state)

    current_epoch = get_current_epoch(test_state)
    validator_index = get_active_validator_indices(test_state.validator_registry, current_epoch)[-1]

    assert test_state.validator_registry[validator_index].exit_epoch == FAR_FUTURE_EPOCH

    # set validator balance to below ejection threshold
    test_state.validator_balances[validator_index] = EJECTION_BALANCE - 1


    #
    # trigger epoch transition
    #
    block = construct_empty_block_for_next_slot(test_state)
    block.slot += SLOTS_PER_EPOCH
    state_transition(test_state, block)

    assert test_state.validator_registry[validator_index].exit_epoch < FAR_FUTURE_EPOCH

    return state, [block], test_state


@timeit
def test_historical_batch(state):
    test_state = deepcopy(state)

    test_state.slot += SLOTS_PER_HISTORICAL_ROOT - (test_state.slot % SLOTS_PER_HISTORICAL_ROOT) - 1
    block = construct_empty_block_for_next_slot(test_state)

    state_transition(test_state, block)

    assert test_state.slot == block.slot
    assert get_current_epoch(test_state) % (SLOTS_PER_HISTORICAL_ROOT // SLOTS_PER_EPOCH) == 0
    assert len(test_state.historical_roots) == len(state.historical_roots) + 1

    return state, [block], test_state
    

@timeit
def sanity_tests():
    print("Buidling state with 100 validators...")
    genesis_state = create_genesis_state(num_validators=100)
    print("done!")
    print()

    test_cases = []

    print("Running some sanity check tests...\n")
    test_slot_transition(genesis_state)
    print("Passed slot transition test\n")
    test_cases.append(test_empty_block_transition(genesis_state))
    print("Passed empty block transition test\n")
    test_cases.append(test_skipped_slots(genesis_state))
    print("Passed skipped slot test\n")
    test_cases.append(test_empty_epoch_transition(genesis_state))
    print("Passed empty epoch transition test\n")
    test_cases.append(test_empty_epoch_transition_not_finalizing(genesis_state))
    print("Passed non-finalizing epoch test\n")
    test_cases.append(test_proposer_slashing(genesis_state))
    print("Passed proposer slashing test\n")
    test_cases.append(test_attestation(genesis_state))
    print("Passed attestation test\n")
    test_cases.append(test_deposit_in_block(genesis_state))
    print("Passed deposit test\n")
    test_cases.append(test_voluntary_exit(genesis_state))
    print("Passed voluntary exit test\n")
    test_cases.append(test_transfer(genesis_state))
    print("Passed transfer test\n")
    test_cases.append(test_ejection(genesis_state))
    print("Passed ejection test\n")
    test_cases.append(test_historical_batch(genesis_state))
    print("Passed historical batch test\n")
    print("done!")

    return test_cases

# Monkey patch validator shuffling cache
_get_shuffling = spec.get_shuffling
shuffling_cache = {}
def get_shuffling(seed: Bytes32,
                  validators: List[Validator],
                  epoch: Epoch) -> List[List[ValidatorIndex]]:
    
    param_hash = (seed, hash_tree_root(validators, [Validator]), epoch)

    if param_hash in shuffling_cache:
        #print("Cache hit, epoch={0}".format(epoch))
        return shuffling_cache[param_hash]
    else:
        #print("Cache miss, epoch={0}".format(epoch))
        ret = _get_shuffling(seed, validators, epoch)
        shuffling_cache[param_hash] = ret
        return ret

spec.get_shuffling = get_shuffling

hash_cache = {}
def hash(x):
    if x in hash_cache:
        return hash_cache[x]
    else:
        ret = sha256(x).digest()
        hash_cache[x] = ret
        return ret

spec.hash = hash

if __name__ == "__main__":
    test_cases = sanity_tests()
    
    if "--generate-json" in sys.argv:
        j = {}
        j["title"] = "Sanity tests"
        j["summary"] = "Basic sanity checks from phase 0 spec pythonization"
        j["test_suite"] = "sanity_tests"
        j["fork"] = "tchaikovsky"
        j["version"] = "1.0"

        test_cases_json = []
        for test_case in test_cases:
            config = {    
                "SHARD_COUNT": spec.SHARD_COUNT,
                "TARGET_COMMITTEE_SIZE": spec.TARGET_COMMITTEE_SIZE,
                "GENESIS_SLOT": spec.GENESIS_SLOT,
                "GENESIS_EPOCH": spec.GENESIS_EPOCH,
                "MIN_ATTESTATION_INCLUSION_DELAY": spec.MIN_ATTESTATION_INCLUSION_DELAY,
                "SLOTS_PER_EPOCH": spec.SLOTS_PER_EPOCH,
                "LATEST_RANDAO_MIXES_LENGTH": spec.LATEST_RANDAO_MIXES_LENGTH,
                "SLOTS_PER_HISTORICAL_ROOT": spec.SLOTS_PER_HISTORICAL_ROOT,
                "LATEST_ACTIVE_INDEX_ROOTS_LENGTH": spec.LATEST_ACTIVE_INDEX_ROOTS_LENGTH,
                "LATEST_SLASHED_EXIT_LENGTH": spec.LATEST_SLASHED_EXIT_LENGTH,
            }
            initial_state = jsonize(test_case[0], type(test_case[0]))
            blocks = jsonize(test_case[1], [type(test_case[1][0])])
            expected_state = jsonize(test_case[2], type(test_case[2]))
            expected_state_root = hash_tree_root(test_case[2], type(test_case[2]))
            test_cases_json.append({
                "config": config,
                "initial_state": initial_state,
                "blocks": blocks,
                "expected_state": expected_state,
                "expected_state_root": expected_state_root.hex(),
            })

        j["test_cases"] = test_cases_json

        with open("test_cases.json", "w") as f:
            json.dump(j, f, indent=4)