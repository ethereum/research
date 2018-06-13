from full_pos import blake, mk_genesis_state
import bls
from simpleserialize import serialize, deserialize, eq, deepcopy

privkeys = [int.from_bytes(blake(str(i).encode('utf-8'))[:5], 'big') for i in range(400)]
print('Generated privkeys')
keymap = {}
for i,k in enumerate(privkeys):
    keymap[bls.privtopub(k)] = k
    if i%50 == 0:
        print("Generated %d keys" % i)

def mock_make_child(parent_state, parent, skips, attester_share=0.8, checkpoint_shards=[]):
    crystallized_state, active_state = parent_state
    parent_attestation_hash = parent_hash
    validator_count = len(parent_state.active_validators)
    attestation_count = min(parent_state.active_validators, 128)
    indices, main_signer = get_attesters_and_signer(crystallized_state, active_state, skips)
    main_signer = indices[-1]
    # Randomly pick indices to include
    bitfield = [1 if random.random() < attester_share else 0 for i in indices]
    # Attestations
    sigs = [bls.sign(parent_attestation_hash, keymap[parent_state.active_validators[indices[i]].pubkey])
            for i in range(len(indices)) if bitfield[i]]
    attestation_aggregate_sig = bls.aggregate_sig(sigs)
    attestation_bitmask = bytearray((len(bitfield)-1) // 8 + 1)
    for i, b in enumerate(bitfield):
        attestation_bitmask[i//8] ^= (128 >> (i % 8)) * b
    # Randomly pick indices to include for checkpoints
    shard_aggregate_votes = []
    for shard, crosslinker_share in checkpoint_shards:
        indices = parent_state.shuffling[(validator_count * shard) // 100: (validator_count * (shard + 1)) // 100]
        bitfield = [1 if random.random() < crosslinker_share else 0 for i in indices]
        bitmask = bytearray((len(bitfield)-1) // 8 + 1)
        checkpoint = blake(bytes([shard]))
        checkpoint_attestation_hash = get_checkpoint_aggvote_msg(shard, checkpoint, crystallized_state)
        sigs = [bls.sign(checkpoint_attestation_hash, keymap[parent_state.active_validators[indices[i]].pubkey])
                for i in range(len(indices)) if bitfield[i]]
        shard_aggregate_votes.append(AggregateVote(shard, checkpoint, bitmask, bls.aggregate_sig(sigs)))
    # State calculations
    o = BlockHeader(parent.hash, skips, blake(str(random.random()).encode('utf-8')),
                       attestation_bitmask, attestation_aggregate_sig, shard_aggregate_votes,
                       b'\x00'*32, b'\x00'*32, state.height)
    new_crystallized_state, new_active_state = compute_state_transition(crystallized_state, active_state, o)
    if crystallized_state == new_crystallized_state:
        o.state_hash = blake(parent.state_hash[:32] + blake(serialize(new_active_state)))
    else:
        o.state_hash = blake(blake(serialize(new_crystallized_state)) + blake(serialize(new_active_state)))
    # Main signature
    o.sign(keymap[parent_state.active_validators[indices[-1]].pubkey])
    return o

c, a = mk_genesis_state(keymap.keys())
print('Generated genesis state')
print('Crystallized state length:', len(serialize(c)))
print('Active state length:', len(serialize(a)))
