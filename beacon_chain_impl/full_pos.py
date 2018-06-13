from hashlib import blake2s
blake = lambda x: blake2s(x).digest()
from rlp.sedes import big_endian_int, Binary, binary, CountableList, BigEndianInt, Binary
int256 = BigEndianInt(256)
hash32 = Binary.fixed_length(32)
import rlp
import bls
import random
from bls import decompress_G1, aggregate_pubs, verify, sign
from simpleserialize import deepcopy


SHARD_COUNT = 100
DEFAULT_BALANCE = 20000

class AggregateVote():
    fields = {
        'shard_id': 'int16',
        'checkpoint_hash': 'hash32',
        'signer_bitmask': 'bytes',
        'aggregate_sig': 'int256'
    }
    defaults = {
        'shard_id': 0,
        'checkpoint_hash': b'\x00'*32,
        'signer_bitmask': b'',
        'aggregate_sig': 0,
    }

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

class BeaconBlock():
    
    fields = {
        # Hash of the parent block
        'parent_hash': 'hash32',
        # Number of skips (for the full PoS mechanism)
        'skip_count': 'int64',
        # Randao commitment reveal
        'randao_reveal': 'hash32',
        # Bitmask of who participated in the block notarization committee
        'attestation_bitmask': 'bytes',
        # Their aggregate sig
        'attestation_aggregate_sig': 'int256',
        # Shard aggregate votes
        'shard_aggregate_votes': [AggregateVote],
        # Reference to main chain block
        'main_chain_ref': 'hash32',
        # Hash of the state
        'state_hash': 'bytes',
        # Block height
        'height': 'int64',
        # Signature from signer
        'sig': 'int256'
    }

    defaults = {
        'parent_hash': b'\x00'*32,
        'skip_count': 0,
        'randao_reveal': b'\x00'*32,
        'attestation_bitmask': b'',
        'attestation_aggregate_sig': 0,
        'shard_aggregate_votes': [],
        'main_chain_ref': b'\x00'*32,
        'state_hash': b'\x00'*32,
        'height': 0,
        'sig': 0
    }

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

    def sign(self, key):
        self.sig = 0
        self.sig = sign(serialize(self), key)

    def verify(self, pub):
        zig = self.sig
        self.sig = 0
        o = verify(serialize(self), pub, zig)
        self.sig = zig
        return o

def get_shuffling(seed, validator_count, sample=None):
    assert validator_count <= 16777216
    rand_max = 16777216 - 16777216 % validator_count
    o = list(range(validator_count)); source = seed
    i = 0
    while i < (sample if sample is not None else validator_count):
        source = blake(source)
        for pos in range(0, 30, 3):
            m = int.from_bytes(source[pos:pos+3], 'big')
            remaining = validator_count - i
            if remaining == 0:
                break
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

    fields = {'shard_id': 'int16', 'checkpoint_hash': 'hash32', 'voter_bitmask': 'bytes'}
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
    fields = {'active_validators': [ValidatorRecord],
               'queued_validators': [ValidatorRecord],
               'exited_validators': [ValidatorRecord],
               'current_shuffling': ['int24'],
               'current_epoch': 'int64',
               'last_justified_epoch': 'int64',
               'last_finalized_epoch': 'int64',
               'dynasty': 'int64',
               'next_shard': 'int16',
               'current_checkpoint': 'hash32',
               'checkpoint_last_crosslinked': ['int64'],
               'total_deposits': 'int256'}
    defaults = {'active_validators': [],
               'queued_validators': [],
               'exited_validators': [],
               'current_shuffling': ['int24'],
               'current_epoch': 0,
               'last_justified_epoch': 0,
               'last_finalized_epoch': 0,
               'dynasty': 0,
               'next_shard': 0,
               'current_checkpoint': b'\x00'*32,
               'checkpoint_last_crosslinked': [],
               'total_deposits': 0}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

def get_checkpoint_aggvote_msg(shard_id, checkpoint_hash, crystallized_state):
    return shard_id.to_bytes(2, 'big') + \
        checkpoint_hash + \
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
    if block.number % SHARD_COUNT == 1:
        # Process rewards from FFG/crosslink votes
        new_validator_records = deepcopy(crystallized_state.active_validators)
        # Who voted in the last epoch
        ffg_voter_bitmask = bytearray(active_state.ffg_voter_bitmask)
        # Total deposit size
        total_deposits = crystallized_state.total_deposits
        # Number of epochs since last finality
        finality_distance = crystallized_state.epoch - crystallized_state.last_finalized_epoch
        online_reward = 3 if finality_distance == 2 else 0
        offline_penalty = 2 * finality_distance
        total_vote_count = 0
        total_vote_deposits = 0
        for i in range(len(active_validators)):
            if ffg_voter_bitmask[i // 8] & (1 << (i % 8)):
                total_vote_deposits += new_validator_records[i].balance
                new_validator_records[i].balance += online_reward
                total_vote_count += 1
            else:
                new_validator_records[i].balance -= offline_penalty
        total_deposits += total_vote_count * online_reward - \
            (len(active_validators) - total_vote_count) * online_penalty
        # Find the most popular crosslink in each shard
        main_crosslink = {}
        for c in active_state.checkpoints:
            vote_count = 0
            mask =  bytearray(c.voter_bitmask)
            for byte in mask:
                for j in range(8):
                    vote_count += (byte >> j) % 2
            if vote_count > main_crosslink.get(c.shard_id, (b'', 0, b''))[1]:
                main_crosslink[c.shard_id] = (c.checkpoint_hash, vote_count, mask)
        # Adjust crosslinks
        new_checkpoint_last_crosslinked = deepcopy(crystallized_state.checkpoint_last_crosslinked)
        for shard in range(SHARD_COUNT):
            h, votes, mask = main_crosslink.get(shard, (b'', 0))
            crosslink_distance = crystallized_state.epoch - crystallized_state.checkpoint_last_crosslinked[shard]
            indices = get_shard_attesters(crystallized_state, shard)
            online_reward = 3 if crosslink_distance <= 2 else 0
            offline_penalty = crosslink_distance * 2
            for i, index in enumerate(indices):
                if mask[i//8] & (1 << (i % 8)):
                    new_validator_records[index].balance += online_reward
                else:
                    new_validator_records[index].balance -= offline_penalty
            # New checkpoint last crosslinked record
            if votes * 3 >= len(indices) * 2:
                new_checkpoint_last_crosslinked[shard] = crystallized_state.epoch

        # Process other balance deltas
        for i in active_state.balance_deltas:
            if i % 256 <= 128:
                new_validator_records[i >> 8] += i % 256
            else:
                new_validator_records[i >> 8] += (i % 256) - 256
        # Process finality and validator set changes
        justify, finalize = False
        if total_vote_deposits * 3 >= total_deposits * 2:
            justify = True
            if crystallized_state.last_justified_epoch == crystallized_state.current_epoch - 1:
                finalize = True
        if finalize:
            new_active_validators = [v for v in crystallized_state.active_validators]
            new_exited_validators = [v for v in crystallized_state.exited_validators]
            i = 0
            while i < len(new_active_validators):
                if new_validator_records[i].balance <= DEFAULT_BALANCE // 2:
                    new_exited_validators.append(new_validator_records.pop(i))
                elif new_validator_records.switch_dynasty == crystallized_state.dynasty + 1:
                    new_exited_validators.append(new_validator_records.pop(i))
                else:
                    i += 1
            induct = min(len(crystallized_state.queued_validators), crystallized_state.active_validators // 30 + 1)
            for i in range(induct):
                new_active_validators.append(crystallized_state.queued_validators[i])
            new_queued_validators = crystallized_state.queued_validators[induct:]
        else:
            new_queued_validators = crystallized_state.queued_validators
            new_active_validators = crystallized_state.active_validators
            new_exited_validators = crystallized_state.exited_validators
        crystallized_state = CrystallizedState(
            queued_validators=new_queued_validators,
            active_validators=new_active_validators,
            exited_validators=new_exited_validators,
            current_shuffling=get_shuffling(active_state.randao, len(new_active_validators)),
            last_justified_epoch = crystallized_state.current_epoch if justified else crystallized_state.last_justified_epoch,
            last_finalized_epoch = crystallized_state.current_epoch-1 if finalized else crystallized_state.last_finalized_epoch,
            dynasty = crystallized_state.dynasty + (1 if finalized else 0),
            next_shard = 0,
            current_epoch = parent_block.hash, 
            checkpoint_last_crosslinked = new_checkpoint_last_crosslinked,
            total_deposits = total_deposits
        )
        # Reset the active state
        active_state = ActiveState(height=active_state.height,
                                   randao=active_state.randao,
                                   ffg_voter_bitmask=bytearray((len(crystallized_state.active_validators) + 7) // 8),
                                   balance_deltas=[],
                                   checkpoints=[],
                                   total_skip_count=active_state.total_skip_count)
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
    checkpoint_votes = {x.checkpoint_hash + x.shard_id.to_bytes(2, 'big'): x.votes for x in active_state.checkpoints}
    new_ffg_bitmask = bytearray(active_state.ffg_voter_bitmask)
    for vote in block.shard_aggregate_votes:
        attestation = get_checkpoint_aggvote_msg(vote.shard_id, vote.checkpoint_hash, crystallized_state)
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
        votekey = vote.checkpoint_hash + vote.shard_id.to_bytes(2, 'big')
        checkpoint_votes[votekey] = checkpoint_votes.get(votekey, 0) + voters
    
    o =  ActiveState(height=active_state.height + 1,
                     randao=(int.from_bytes(active_state.randao, 'big') ^ 
                             int.from_bytes(block.randao_reveal)).to_bytes(32, 'big'),
                     total_skip_count=active_state.total_skip_count + block.skip_count,
                     checkpoints=[CheckpointRecord(shard_id=int.from_bytes(h[32:], 'big'),
                                  checkpoint_hash=h[:32], votes=checkpoint_votes[h])
                                  for h in sorted(checkpoint_votes.keys())],
                     ffg_voter_bitmask=new_ffg_bitmask,
                     balance_deltas=active_state.balance_deltas + balance_deltas)
                       
    return crystallized_state, o

def mk_genesis_state(pubkeys):
    c =  CrystallizedState(
        active_validators=[ValidatorRecord(
            pubkey=pub,
            return_shard=0,
            return_address=blake(pub.to_bytes(32, 'big'))[-20:],
            randao_commitment=b'\x55'*32,
            balance=DEFAULT_BALANCE,
            switch_dynasty=9999999999999999999
        ) for pub in pubkeys],
        queued_validators=[],
        exited_validators=[],
        current_shuffling=get_shuffling(b'\x35'*32, len(pubkeys)),
        current_epoch=1,
        last_justified_epoch=0,
        last_finalized_epoch=0,
        dynasty=1,
        next_shard=0,
        current_checkpoint=blake(b'insert EOS constitution here'),
        checkpoint_last_crosslinked=[0] * SHARD_COUNT,
        total_deposits=DEFAULT_BALANCE*len(pubkeys))
    a = ActiveState(height=0,
        randao=b'\x45'*32,
        ffg_voter_bitmask=bytearray((len(c.active_validators) + 7) // 8),
        balance_deltas=[],
        checkpoints=[],
        total_skip_count=0)
    return c, a
