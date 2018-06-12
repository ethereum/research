from hashlib import blake2s
blake = lambda x: blake2s(x).digest()
from rlp.sedes import big_endian_int, Binary, binary, CountableList, BigEndianInt, Binary
int256 = BigEndianInt(256)
hash32 = Binary.fixed_length(32)
import rlp
import bls
import random
from bls import decompress_G1, aggregate_pubs, verify

privkeys = [int.from_bytes(blake(str(i).encode('utf-8')), 'big') for i in range(30)]
keymap = {bls.privtopub(k): k for k in privkeys}

SHARD_COUNT = 100

class AggregateVote(rlp.Serializable):
    fields = [
        ('shard_id', int256),
        ('checkpoint_hash', hash32),
        ('signer_bitmask', binary),
        ('aggregate_sig', int256)
    ]

    def __init__(self, shard_id, checkpoint_hash, signer_bitmask, aggregate_sig):
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

    fields = {'checkpoint_hash': 'hash32', 'voters': 'int16'}
    defaults = {}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))


class ActiveState():

    fields = {'height': 'int64', 'randao': 'hash32',
            'ffg_voter_bitmask': 'bytes', 'balance_deltas': ['int32'],
            'checkpoints': [CheckpointRecord],
            'total_skip_count': 'int64'}
    defaults = {'height': 0, 'randao': b'\x00'*32,
        'ffg_voter_bitmask': b'', 'balance_deltas': [],
        'checkpoints': [], 'total_skip_count': 0}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))
                                        
class CrystallizedState():
    def __init__(self, **kwargs):
        self.fields = {'active_validators': [ValidatorRecord],
                       'queued_validators': [ValidatorRecord],
                       'exited_validators': [ValidatorRecord],
                       'current_shuffling': ['int24'],
                       'current_epoch': 'int64',
                       'last_justified_epoch': 'int64',
                       'last_finalized_epoch': 'int64',
                       'dynasty': 'int64',
                       'current_checkpoint': 'hash32',
                       'total_deposits': 'int256'}
        self.defaults = {'active_validators': [],
                       'queued_validators': [],
                       'exited_validators': [],
                       'current_shuffling': ['int24'],
                       'current_epoch': 0,
                       'last_justified_epoch': 0,
                       'last_finalized_epoch': 0,
                       'dynasty': 0,
                       'current_checkpoint': b'\x00'*32,
                       'total_deposits': 0}
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

def get_checkpoint_aggvote_msg(aggvote, crystallized_state):
    return aggvote.shard_id.to_bytes(2, 'big') + \
        aggvote.checkpoint_hash + \
        crystallized_state.current_checkpoint + \
        crystallized_state.current_epoch.to_bytes(8, 'big') + \
        crystallized_state.last_justified_epoch.to_bytes(8, 'big')

def get_attesters_and_signer(crystallized_state, active_state, skip_count):
    attestation_count = min(crystallized_state.active_validators, 128)
    indices = get_shuffling(active_state.randao, len(crystallized_state.active_validators),
                            attestation_count + skip_count + 1)
    return indices[:attestation_count], indices[-1]

def get_shard_attesters(crystallized_state, shard_id):
    vc = len(crystallized_state.active_validators)
    return crystallized_state.current_shuffling[(vc * shard_id) // SHARD_COUNT: (vc * (shard_id + 1)) // SHARD_COUNT]

def compute_state_transition(parent_state, parent_block, block):
    crystallized_state, active_state = parent_state
    # Possibly initialize a new epoch
    # Process the block-by-block stuff

    # Verify the attestations of the parent
    attestation_indices, main_signer = \
        get_attesters_and_signer(crystallized_state, active_state, block.skip_count)
    pubs = []
    balance_deltas = []
    assert len(block.attestation_bitmask) == len(attestation_indices + 7) // 8
    for i, index in enumerate(attestation_indices):
        if block.attestation_bitmask[i//8] & (1<<(i%8)):
            pubs.append(crystallized_state.active_validators[index])
            balance_deltas.append((index << 8) + 1)
    assert len(balance_deltas) <= 128
    balance_deltas.append((main_signer << 8) + len(balance_deltas))
    assert verify(parent_block.hash, aggregate_pubs(pubs), block.aggregate_sig)

    # Verify the attestations of checkpoint hashes
    checkpoint_votes = {x.checkpoint_hash: x.votes for x in active_state.checkpoints}
    new_ffg_bitmask = bytearray(active_state.ffg_voter_bitmask)
    for vote in block.shard_aggregate_votes:
        attestation = get_checkpoint_aggvote_msg(vote, crystallized_state)
        indices = get_shard_attesters(crystallized_state, vote.shard_id)
        assert len(vote.signer_bitmask) == len(indices + 7) // 8
        pubs = []
        voters = 0
        for i, index in enumerate(indices):
            if (vote.signer_bitmask[i//8] >> (i%8)) % 2:
                pubs.append(crystallized_state.active_validators[index])
                if new_ffg_bitmask[index//8] & (1<<(index%8)) == 0:
                    new_ffg_bitmask[index//8] ^= 1<<(index%8)
                    voters += 1
        assert verify(attestation, aggregate_pubs(pubs), vote.aggregate_sig)
        balance_deltas.append((main_signer << 8) + (voters * 16 // len(indices)))
        checkpoint_votes[vote.checkpoint_hash] = checkpoint_votes.get(vote.checkpoint_hash, 0) + voters
    
        
    o =  ActiveState(height=active_state.height + 1,
                     randao=(int.from_bytes(active_state.randao, 'big') ^ 
                             int.from_bytes(block.randao_reveal)).to_bytes(32, 'big'),
                     total_skip_count=active_state.total_skip_count + block.skip_count,
                     checkpoints=[CheckpointRecord(checkpoint_hash=h, votes=checkpoint_votes[h])
                                  for h in sorted(checkpoint_votes.keys())],
                     ffg_voter_bitmask=new_ffg_bitmask,
                     balance_deltas=active_state.balance_deltas + balance_deltas)
                       
                       


def mock_make_child(parent_state, parent_hash, skips, attester_share=0.8, checkpoint_shards=[]):
    parent_attestation_hash = parent_hash
    validator_count = len(parent_state.active_validators)
    attestation_count = min(parent_state.active_validators, 128)
    indices = get_shuffling(parent.randao_state, validator_count,
                            attestation_count + skip_count + 1)
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
        checkpoint_attestation_hash = checkpoint + \
                                  parent_state.checkpoint_hash + \
                                  parent_state.epoch.to_bytes(32, 'big') + \
                                  parent_state.source_epoch.to_bytes(32, 'big')
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
