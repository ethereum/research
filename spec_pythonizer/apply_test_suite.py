import sys
import json
from jsonize import dejsonize
import spec
from copy import deepcopy

if len(sys.argv) != 2:
    print("Usage: python apply_test_suite json_file.json")
    sys.exit(1)

with open(sys.argv[1], "rb") as f:
    j = json.load(f)

print("Loaded test suite {0}".format(j["title"]))
print("Summary: {0}".format(j["summary"]))
print("Test suite: {0[test_suite]}, fork: {0[fork]}, version: {0[version]}".format(j))

def apply_test(config, initial_state, blocks, expected_state, expected_state_root):
    print("Applying config")
    print(json.dumps(config, indent=4))
    for key, value in config.items():
        setattr(spec, key, value)
    state = deepcopy(initial_state)
    for block in blocks:
        spec.state_transition(state, block)

    assert state == expected_state
    if expected_state_root:
        assert spec.hash_tree_root(state) == expected_state_root

    print("Test passed\n")

for i, test_case in enumerate(j["test_cases"]):
    config = test_case["config"]
    initial_state = dejsonize(test_case["initial_state"], spec.BeaconState)
    blocks = dejsonize(test_case["blocks"], [spec.BeaconBlock])
    expected_state = dejsonize(test_case["expected_state"], spec.BeaconState)
    expected_state_root = bytes.fromhex(test_case["expected_state_root"]) if "expected_state_root" in test_case else None

    print("Running test {0}".format(i))
    apply_test(config, initial_state, blocks, expected_state, expected_state_root)