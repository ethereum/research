import spec
from spec import (
    hash_tree_root,
)

from jsonize import jsonize

CONFIG_FIELDS = [
    # Misc
    "SHARD_COUNT",
    "TARGET_COMMITTEE_SIZE",
    "MAX_BALANCE_CHURN_QUOTIENT",
    "MAX_INDICES_PER_SLASHABLE_VOTE",
    "MAX_EXIT_DEQUEUES_PER_EPOCH",
    "SHUFFLE_ROUND_COUNT",
    # Deposit contract
    "DEPOSIT_CONTRACT_TREE_DEPTH",
    # Gwei values
    "MIN_DEPOSIT_AMOUNT",
    "MAX_DEPOSIT_AMOUNT",
    "FORK_CHOICE_BALANCE_INCREMENT",
    "EJECTION_BALANCE",
    # Initial values
    "GENESIS_FORK_VERSION",
    "GENESIS_SLOT",
    "GENESIS_EPOCH",
    "GENESIS_START_SHARD",
    "BLS_WITHDRAWAL_PREFIX_BYTE",
    # Time parameters
    "SECONDS_PER_SLOT",
    "MIN_ATTESTATION_INCLUSION_DELAY",
    "SLOTS_PER_EPOCH",
    "MIN_SEED_LOOKAHEAD",
    "ACTIVATION_EXIT_DELAY",
    "EPOCHS_PER_ETH1_VOTING_PERIOD",
    "SLOTS_PER_HISTORICAL_ROOT",
    "MIN_VALIDATOR_WITHDRAWABILITY_DELAY",
    "PERSISTENT_COMMITTEE_PERIOD",
    # State list lengths
    "LATEST_RANDAO_MIXES_LENGTH",
    "LATEST_ACTIVE_INDEX_ROOTS_LENGTH",
    "LATEST_SLASHED_EXIT_LENGTH",
    # Reward and penalties
    "BASE_REWARD_QUOTIENT",
    "WHISTLEBLOWER_REWARD_QUOTIENT",
    "ATTESTATION_INCLUSION_REWARD_QUOTIENT",
    "INACTIVITY_PENALTY_QUOTIENT",
    "MIN_PENALTY_QUOTIENT",
    # Max transactions per block
    "MAX_PROPOSER_SLASHINGS",
    "MAX_ATTESTER_SLASHINGS",
    "MAX_ATTESTATIONS",
    "MAX_DEPOSITS",
    "MAX_VOLUNTARY_EXITS",
    "MAX_TRANSFERS",
    # Signature domains
    "DOMAIN_BEACON_BLOCK",
    "DOMAIN_RANDAO",
    "DOMAIN_ATTESTATION",
    "DOMAIN_DEPOSIT",
    "DOMAIN_VOLUNTARY_EXIT",
    "DOMAIN_TRANSFER",
]


def generate_test_case(pre_state, blocks, post_state, name=None, config=None, fields=None, verify_signatures=False, root=False):
    if config is None:
        config = {}
    if fields is None:
        fields = []

    test_case = {}
    test_config = {}
    for field in CONFIG_FIELDS:
        if field in config:
            value = config[field]
        else:
            value = getattr(spec, field)
        test_config[field] = '0x' + value.hex() if isinstance(value, bytes) else value
    if name:
        test_case['name'] = name
    test_case['config'] = test_config
    test_case['verify_signatures'] = verify_signatures
    test_case['initial_state'] = jsonize(pre_state, type(pre_state))
    test_case['blocks'] = jsonize(blocks, [type(blocks[0])])
    if len(fields) == 0:
        test_case['expected_state'] = jsonize(post_state, type(post_state))
    else:
        # only support for top level fields
        test_case['expected_state'] = {
            field: jsonize(getattr(post_state, field), post_state.fields[field])
            for field in fields
        }
    if root:
        test_case['expected_state_root'] = hash_tree_root(post_state, type(post_state)).hex()

    return test_case


def generate_from_test(test_fn, init_state, config=None, fields=None, verify_signatures=False, root=False):
    #
    # ``test_fn`` must accept ``start_state`` and return (blocks, post_state)
    #
    if config is None:
        config = {}
    if fields is None:
        fields = []

    pre_state, blocks, post_state = test_fn(init_state)
    return generate_test_case(
        pre_state,
        blocks,
        post_state,
        name=test_fn.__name__,
        config=config,
        verify_signatures=verify_signatures,
        fields=fields
    )


def dump_yaml(test, outfile):
    import oyaml as yaml

    yaml.safe_dump(test['metadata'], outfile, default_flow_style=False)
    yaml.safe_dump({'test_cases': test['test_cases']}, outfile, default_flow_style=False)


def dump_json(test, outfile):
    import json

    out = test['metadata'].copy()
    out['test_cases'] = test['test_cases']
    json.dump(out, outfile, indent=4)
