from full_pos import blake, mk_genesis_state_and_block, compute_state_transition, \
    get_attesters_and_signer, Block, get_crosslink_aggvote_msg, AggregateVote, \
    SHARD_COUNT, ATTESTER_COUNT, get_shard_attesters
import random
import bls
from simpleserialize import serialize, deserialize, eq, deepcopy
import time

privkeys = [int.from_bytes(blake(str(i).encode('utf-8'))[:4], 'big') for i in range(100000)]
print('Generated privkeys')
keymap = {}
for i,k in enumerate(privkeys):
    keymap[bls.privtopub(k)] = k
    if i%50 == 0:
        print("Generated %d keys" % i)

def mock_make_child(parent_state, parent, skips, attester_share=0.8, crosslink_shards=[]):
    crystallized_state, active_state = parent_state
    parent_attestation = serialize(parent)
    validator_count = len(crystallized_state.active_validators)
    indices, main_signer = get_attesters_and_signer(crystallized_state, active_state, skips)
    print('Selected indices: %r' % indices)
    print('Selected main signer: %d' % main_signer)
    # Randomly pick indices to include
    bitfield = [1 if random.random() < attester_share else 0 for i in indices]
    # Attestations
    sigs = [bls.sign(parent_attestation, keymap[crystallized_state.active_validators[indices[i]].pubkey])
            for i in range(len(indices)) if bitfield[i]]
    attestation_aggregate_sig = bls.aggregate_sigs(sigs)
    print('Aggregated sig')
    attestation_bitmask = bytearray((len(bitfield)-1) // 8 + 1)
    for i, b in enumerate(bitfield):
        attestation_bitmask[i//8] ^= (128 >> (i % 8)) * b
    print('Aggregate bitmask:', bin(int.from_bytes(attestation_bitmask, 'big')))
    # Randomly pick indices to include for crosslinks
    shard_aggregate_votes = []
    for shard, crosslinker_share in crosslink_shards:
        print('Making crosslink in shard %d' % shard)
        indices = get_shard_attesters(crystallized_state, shard)
        print('Indices: %r' % indices)
        bitfield = [1 if random.random() < crosslinker_share else 0 for i in indices]
        bitmask = bytearray((len(bitfield)+7) // 8)
        for i, b in enumerate(bitfield):
            bitmask[i//8] ^= (128 >> (i % 8)) * b
        print('Bitmask:', bin(int.from_bytes(bitmask, 'big')))
        shard_block_hash = blake(bytes([shard]))
        crosslink_attestation_hash = get_crosslink_aggvote_msg(shard, shard_block_hash, crystallized_state)
        sigs = [bls.sign(crosslink_attestation_hash, keymap[crystallized_state.active_validators[indices[i]].pubkey])
                for i in range(len(indices)) if bitfield[i]]
        v = AggregateVote(shard_id=shard,
                          shard_block_hash=shard_block_hash,
                          signer_bitmask=bitmask,
                          aggregate_sig=list(bls.aggregate_sigs(sigs)))
        shard_aggregate_votes.append(v)
    print('Added %d shard aggregate votes' % len(crosslink_shards))
    # State calculations
    o = Block(parent_hash=blake(parent_attestation),
              skip_count=skips,
              randao_reveal=blake(str(random.random()).encode('utf-8')),
              attestation_bitmask=attestation_bitmask,
              attestation_aggregate_sig=list(attestation_aggregate_sig),
              shard_aggregate_votes=shard_aggregate_votes,
              main_chain_ref=b'\x00'*32,
              state_hash=b'\x00'*64)
    print('Generated preliminary block header')
    new_crystallized_state, new_active_state = \
        compute_state_transition((crystallized_state, active_state), parent, o, verify_sig=False)
    print('Calculated state transition')
    if crystallized_state == new_crystallized_state:
        o.state_hash = blake(parent.state_hash[:32] + blake(serialize(new_active_state)))
    else:
        o.state_hash = blake(blake(serialize(new_crystallized_state)) + blake(serialize(new_active_state)))
    # Main signature
    o.sign(keymap[crystallized_state.active_validators[main_signer].pubkey])
    print('Signed')
    return o, new_crystallized_state, new_active_state

c, a, block = mk_genesis_state_and_block(keymap.keys())
print('Generated genesis state')
print('Crystallized state length:', len(serialize(c)))
print('Active state length:', len(serialize(a)))
print('Block size:', len(serialize(block)))
block2, c2, a2 = mock_make_child((c, a), block, 0, 0.8, [])
t = time.time()
assert compute_state_transition((c, a), block, block2)
print("Normal block (basic attestation only) processed in %.4f sec" % (time.time() - t))
print('Verified a block!')
block3, c3, a3 = mock_make_child((c2, a2), block2, 0, 0.8, [(0, 0.75)])
print('Verified a block with a committee!')
while a3.height % SHARD_COUNT > 0:
    block3, c3, a3 = mock_make_child((c3, a3), block3, 0, 0.8, [(a3.height, 0.6 + 0.02 * a3.height)])
    print('Height: %d' % a3.height)
print('FFG bitmask:', bin(int.from_bytes(a3.ffg_voter_bitmask, 'big')))
block4, c4, a4 = mock_make_child((c3, a3), block3, 1, 0.55, [])
t = time.time()
assert compute_state_transition((c3, a3), block3, block4)
print("Epoch transition processed in %.4f sec" % (time.time() - t))
