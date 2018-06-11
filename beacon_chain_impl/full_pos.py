from hashlib import blake2s
blake = lambda x: blake2s(x).digest()
from rlp.sedes import big_endian_int, Binary, binary, CountableList, BigEndianInt, Binary
int256 = BigEndianInt(256)
hash32 = Binary.fixed_length(32)
import rlp
import bls
import random

privkeys = [int.from_bytes(blake(str(i).encode('utf-8')), 'big') for i in range(30)]
keymap = {bls.privtopub(k): k for k in privkeys}

class AggregateVote(rlp.Serializable):
    fields = [
        ('shard_id', int256),
        ('checkpoint', hash32),
        ('signer_bitmask', binary),
        ('aggregate_sig', int256)
    ]

    def __init__(self, shard_id, checkpoint, signer_bitmask, aggregate_sig):
        # at the beginning of a method, locals() is a dict of all arguments
        fields = {k: v for k, v in locals().items() if k != 'self'}
        super(BlockHeader, self).__init__(**fields)

class BeaconBlock(rlp.Serializable):
    
    fields = [
        # Hash of the parent block
        ('parent_hash', hash32),
        # Number of skips (for the full PoS mechanism)
        ('skip_count', int256),
        # Randao commitment reveal
        ('randao_reveal', hash32),
        # Bitmask of who participated in the block notarization committee
        ('attestation_bitmask', binary),
        # Their aggregate sig
        ('attestation_aggregate_sig', int256),
        # Shard aggregate votes
        ('shard_aggregate_votes', CountableList(AggregateVote)),
        # Reference to main chain block
        ('main_chain_ref', hash32),
        # Hash of the state
        ('state_hash', hash32),
        # Block height
        ('height', int256),
        # Signature from signer
        ('sig', int256)
    ]

    def __init__(self,
                 parent_hash=b'\x00'*32, skip_count=0, randao_reveal=b'\x00'*32,
                 attestation_bitmask=b'', attestation_aggregate_sig=0,
                 shard_aggregate_votes=[], main_chain_ref=b'\x00'*32,
                 state_hash=b'\x00'*32, height=0, sig=0):
        # at the beginning of a method, locals() is a dict of all arguments
        fields = {k: v for k, v in locals().items() if k != 'self'}
        super(BlockHeader, self).__init__(**fields)

def get_shuffling(seed, validator_count, sample=None):
    assert validator_count <= 16777216
    rand_max = 16777216 - 16777216 % validator_count
    o = list(range(validator_count)); source = seed
    i = 0
    while i < sample if sample is not None else validator_count:
        source = blake(source)
        for pos in range(0, 30, 3):
            m = int.from_bytes(source[pos:pos+3], 'big')
            remaining = validator_count - i
            if validator_count < rand_max:
                replacement_pos = m % remaining + i
                o[i], o[replacement_pos] = o[replacement_pos], o[i]
                i += 1
    return o

class ValidatorRecord():
    fields = {'pubkey': 'int256', 'return_shard': 'int16',
               'return_address': 'address', 'randao_commitment': 'hash32',
               'balance': 'int64', 'switch_dynasty': 'int64'}
    defaults = {}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

class CheckpointRecord():

    fields = {'checkpoint_hash': 'hash32', 'bitmask': 'bytes'}
    defaults = {}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))


class ActiveState():

    fields = {'height': 'int64', 'randao': 'hash32',
            'validator_ffg_voted': 'bytes', 'rewarded': ['int24'],
            'penalized': ['int24'], 'checkpoints': [CheckpointRecord],
            'total_skip_count': 'int64'}
    defaults = {'height': 0, 'randao': b'\x00'*32,
        'validator_ffg_voted': b'', 'rewarded': [],
        'penalized': [], 'checkpoints': [], 'total_skip_count': 0}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))
                                        
class CrystallizedState():
    def __init__(self, **kwargs):
        self.fields = {'active_validators': [ValidatorRecord],
                       'queued_validators': [ValidatorRecord],
                       'exited_validators': [ValidatorRecord],
                       'last_justified_epoch': 'int64',
                       'last_finalized_epoch': 'int64',
                       'dynasty': 'int64',
                       'current_checkpoint': 'hash32',
                       'total_deposits': 'int256'}
        self.defaults = {'active_validators': [],
                       'queued_validators': [],
                       'exited_validators': [],
                       'last_justified_epoch': 0,
                       'last_finalized_epoch': 0,
                       'dynasty': 0,
                       'current_checkpoint': b'\x00'*32,
                       'total_deposits': 0}
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))
                 

def compute_state_transition(parent_state, block, ):
    pass

def mock_make_child(parent_state, parent_hash, skips, attester_share=0.8, checkpoint_shards=[]):
    parent_attestation_hash = parent_hash + \
                              parent_state.checkpoint_hash + \
                              parent_state.epoch.to_bytes(32, 'big') + \
                              parent_state.source_epoch.to_bytes(32, 'big')
    checkpoint_attestation_hash = b'\x00' * 32 + \
                              parent_state.checkpoint_hash + \
                              parent_state.epoch.to_bytes(32, 'big') + \
                              parent_state.source_epoch.to_bytes(32, 'big')
    validator_count = len(parent_state.active_validators)
    indices = get_shuffling(parent.randao_state, validator_count,
                            min(parent_state.active_validators, 128))
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
        sigs = [bls.sign(checkpoint, keymap[parent_state.active_validators[indices[i]].pubkey])
                for i in range(len(indices)) if bitfield[i]]
        shard_aggregate_votes.append(AggregateVote(shard, checkpoint, bitmask, bls.aggregate_sig(sigs)))
    # State calculations
    
    o = BlockHeader(parent.hash, skips, blake(str(random.random()).encode('utf-8')),
                       attestation_bitmask, attestation_aggregate_sig, shard_aggregate_votes,
                       b'\x00'*32, state.hash, state.height)
    # Main signature
    o.sign(keymap[parent_state.active_validators[indices[-1]].pubkey])
    return o
