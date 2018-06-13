try:
    from hashlib import blake2s
except:
    from pyblake2 import blake2s
blake = lambda x: blake2s(x).digest()
import bls
import random
from bls import decompress_G1, aggregate_pubs, verify, sign, privtopub
from simpleserialize import deepcopy, serialize, to_dict


SHARD_COUNT = 20
ATTESTER_COUNT = 32
DEFAULT_BALANCE = 20000

class AggregateVote():
    fields = {
        'shard_id': 'int16',
        'checkpoint_hash': 'hash32',
        'signer_bitmask': 'bytes',
        'aggregate_sig': ['int256']
    }
    defaults = {
        'shard_id': 0,
        'checkpoint_hash': b'\x00'*32,
        'signer_bitmask': b'',
        'aggregate_sig': [0,0],
    }

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

class Block():
    
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
        'attestation_aggregate_sig': ['int256'],
        # Shard aggregate votes
        'shard_aggregate_votes': [AggregateVote],
        # Reference to main chain block
        'main_chain_ref': 'hash32',
        # Hash of the state
        'state_hash': 'bytes',
        # Signature from signer
        'sig': ['int256']
    }

    defaults = {
        'parent_hash': b'\x00'*32,
        'skip_count': 0,
        'randao_reveal': b'\x00'*32,
        'attestation_bitmask': b'',
        'attestation_aggregate_sig': [0,0],
        'shard_aggregate_votes': [],
        'main_chain_ref': b'\x00'*32,
        'state_hash': b'\x00'*32,
        'sig': [0,0]
    }

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

    def sign(self, key):
        self.sig = [0,0]
        self.sig = list(sign(serialize(self), key))

    def verify(self, pub):
        zig = self.sig
        self.sig = [0,0]
        o = verify(serialize(self), pub, tuple(zig))
        self.sig = zig
        return o

    @property
    def hash(self):
        return blake(serialize(self))

def get_shuffling(seed, validator_count, sample=None):
    assert validator_count <= 16777216
    rand_max = 16777216 - 16777216 % validator_count
    o = list(range(validator_count)); source = seed
    i = 0
    maxvalue = sample if sample is not None else validator_count
    while i < maxvalue:
        source = blake(source)
        for pos in range(0, 30, 3):
            m = int.from_bytes(source[pos:pos+3], 'big')
            remaining = validator_count - i
            if remaining == 0:
                break
            if validator_count < rand_max:
                replacement_pos = (m % remaining) + i
                o[i], o[replacement_pos] = o[replacement_pos], o[i]
                i += 1
    return o[:maxvalue]

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
            assert k in kwargs or k in self.defaults, k
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

class CrosslinkRecord():
    fields = {'epoch': 'int64', 'hash': 'hash32'}
    defaults = {'epoch': 0, 'hash': b'\x00'*32}

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
               'crosslink_records': [CrosslinkRecord],
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
               'crosslink_records': [],
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
    attestation_count = min(len(crystallized_state.active_validators), ATTESTER_COUNT)
    indices = get_shuffling(active_state.randao, len(crystallized_state.active_validators),
                            attestation_count + skip_count + 1)
    return indices[:attestation_count], indices[-1]

def get_shard_attesters(crystallized_state, shard_id):
    vc = len(crystallized_state.active_validators)
    return crystallized_state.current_shuffling[(vc * shard_id) // SHARD_COUNT: (vc * (shard_id + 1)) // SHARD_COUNT]

def compute_state_transition(parent_state, parent_block, block, verify_sig=True):
    crystallized_state, active_state = parent_state
    # Initialize a new epoch if needed
    if active_state.height % SHARD_COUNT == 0:
        print('Processing epoch transition')
        # Process rewards from FFG/crosslink votes
        new_validator_records = deepcopy(crystallized_state.active_validators)
        # Who voted in the last epoch
        ffg_voter_bitmask = bytearray(active_state.ffg_voter_bitmask)
        # Total deposit size
        total_deposits = crystallized_state.total_deposits
        # Old total deposit size
        td = total_deposits
        # Number of epochs since last finality
        finality_distance = crystallized_state.current_epoch - crystallized_state.last_finalized_epoch
        online_reward = 6 if finality_distance <= 2 else 0
        offline_penalty = 3 * finality_distance
        total_vote_count = 0
        total_vote_deposits = 0
        total_validators = len(crystallized_state.active_validators)
        for i in range(total_validators):
            if ffg_voter_bitmask[i // 8] & (128 >> (i % 8)):
                total_vote_deposits += new_validator_records[i].balance
                new_validator_records[i].balance += online_reward
                total_vote_count += 1
            else:
                new_validator_records[i].balance -= offline_penalty
        print('Total voted: %d of %d validators (%.2f%%), %d of %d deposits (%.2f%%)' %
              (total_vote_count, total_validators, total_vote_count * 100 / total_validators,
               total_vote_deposits, total_deposits, total_vote_deposits * 100 / total_deposits))
        print('FFG online reward: %d, offline penalty: %d' % (online_reward, offline_penalty))
        total_deposits += total_vote_count * online_reward - \
            (total_validators - total_vote_count) * offline_penalty
        print('Total deposit change from FFG: %d' % (total_deposits - td))
        td = total_deposits
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
        new_crosslink_records = deepcopy(crystallized_state.crosslink_records)
        for shard in range(SHARD_COUNT):
            print('Processing crosslink data for shard %d' % shard)
            indices = get_shard_attesters(crystallized_state, shard)
            h, votes, mask = main_crosslink.get(shard, (b'', 0, bytearray((len(indices)+7)//8)))
            crosslink_distance = crystallized_state.current_epoch - crystallized_state.crosslink_records[shard].epoch
            print('Last crosslink from this shard was from epoch %d' % crystallized_state.crosslink_records[shard].epoch)
            online_reward = 3 if crosslink_distance <= 2 else 0
            offline_penalty = crosslink_distance * 2
            for i, index in enumerate(indices):
                if mask[i//8] & (1 << (i % 8)):
                    new_validator_records[index].balance += online_reward
                else:
                    new_validator_records[index].balance -= offline_penalty
            total_deposits += votes * online_reward - (len(indices) - votes) * offline_penalty
            print('Total voters: %d of %d (%.2f%%)' % (votes, len(indices), votes * 100 / len(indices)))
            print('Crosslink online reward: %d, offline penalty: %d' % (online_reward, offline_penalty))
            # New checkpoint last crosslinked record
            if votes * 3 >= len(indices) * 2:
                new_crosslink_records[shard] = CrosslinkRecord(hash=h, epoch=crystallized_state.current_epoch)
                print('Finalized checkpoint: %s' % hex(int.from_bytes(h, 'big')))
        print('Total deposit change from crosslinks: %d' % (total_deposits - td))
        td = total_deposits
        # Process other balance deltas
        for i in active_state.balance_deltas:
            if i % 256 <= 128:
                new_validator_records[i >> 8].balance += i % 256
                total_deposits += i % 256
            else:
                new_validator_records[i >> 8].balance += (i % 256) - 256
                total_deposits += (i % 256) - 256
        print('Total deposit change from deltas: %d' % (total_deposits - td))
        print('New total deposits: %d' % total_deposits)
        # Process finality and validator set changes
        justify, finalize = False, False
        if total_vote_deposits * 3 >= total_deposits * 2:
            justify = True
            print('Justifying last epoch')
            if crystallized_state.last_justified_epoch == crystallized_state.current_epoch - 1:
                finalize = True
                print('Finalizing last epoch')
        if finalize:
            new_active_validators = [v for v in crystallized_state.active_validators]
            new_exited_validators = [v for v in crystallized_state.exited_validators]
            i = 0
            while i < len(new_active_validators):
                if new_validator_records[i].balance <= DEFAULT_BALANCE // 2:
                    new_exited_validators.append(new_validator_records.pop(i))
                elif new_validator_records[i].switch_dynasty == crystallized_state.dynasty + 1:
                    new_exited_validators.append(new_validator_records.pop(i))
                else:
                    i += 1
            induct = min(len(crystallized_state.queued_validators), len(crystallized_state.active_validators) // 30 + 1)
            for i in range(induct):
                if crystallized_state.queued_validators[i].switch_dynasty > crystallized_state.dynasty + 1:
                    induct = i
                    break
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
            last_justified_epoch = crystallized_state.current_epoch if justify else crystallized_state.last_justified_epoch,
            last_finalized_epoch = crystallized_state.current_epoch-1 if finalize else crystallized_state.last_finalized_epoch,
            dynasty = crystallized_state.dynasty + (1 if finalize else 0),
            next_shard = 0,
            current_epoch = crystallized_state.current_epoch + 1,
            crosslink_records = new_crosslink_records,
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
    assert len(block.attestation_bitmask) == (len(attestation_indices) + 7) // 8
    for i, index in enumerate(attestation_indices):
        if block.attestation_bitmask[i//8] & (128>>(i%8)):
            pubs.append(crystallized_state.active_validators[index].pubkey)
            balance_deltas.append((index << 8) + 1)
    assert len(balance_deltas) <= 128
    balance_deltas.append((main_signer << 8) + len(balance_deltas))
    assert verify(parent_block.hash, aggregate_pubs(pubs), block.attestation_aggregate_sig)
    print('Verified aggregate sig')

    # Verify the attestations of checkpoint hashes
    checkpoint_votes = {vote.checkpoint_hash + vote.shard_id.to_bytes(2, 'big'):
                        vote.voter_bitmask for vote in active_state.checkpoints} 
    new_ffg_bitmask = bytearray(active_state.ffg_voter_bitmask)
    for vote in block.shard_aggregate_votes:
        attestation = get_checkpoint_aggvote_msg(vote.shard_id, vote.checkpoint_hash, crystallized_state)
        indices = get_shard_attesters(crystallized_state, vote.shard_id)
        votekey = vote.checkpoint_hash + vote.shard_id.to_bytes(2, 'big')
        if votekey not in checkpoint_votes:
            checkpoint_votes[votekey] = bytearray((len(indices) + 7) // 8)
        bitmask = checkpoint_votes[votekey]
        pubs = []
        voters = 0
        for i, index in enumerate(indices):
            if vote.signer_bitmask[i//8] & (128>>(i%8)):
                pubs.append(crystallized_state.active_validators[index].pubkey)
                if new_ffg_bitmask[index//8] & (128>>(index%8)) == 0:
                    new_ffg_bitmask[index//8] ^= 128>>(index%8)
                    bitmask[i//8] ^= 128>>(i%8)
                    voters += 1
        assert verify(attestation, aggregate_pubs(pubs), vote.aggregate_sig)
        balance_deltas.append((main_signer << 8) + (voters * 16 // len(indices)))
        checkpoint_votes[votekey] = bitmask
        print('Verified aggregate vote')
    
    o =  ActiveState(height=active_state.height + 1,
                     randao=(int.from_bytes(active_state.randao, 'big') ^ 
                             int.from_bytes(block.randao_reveal, 'big')).to_bytes(32, 'big'),
                     total_skip_count=active_state.total_skip_count + block.skip_count,
                     checkpoints=[CheckpointRecord(shard_id=int.from_bytes(h[32:], 'big'),
                                  checkpoint_hash=h[:32], voter_bitmask=checkpoint_votes[h])
                                  for h in sorted(checkpoint_votes.keys())],
                     ffg_voter_bitmask=new_ffg_bitmask,
                     balance_deltas=active_state.balance_deltas + balance_deltas)

    if verify_sig:
        assert block.verify(crystallized_state.active_validators[main_signer].pubkey)
        print('Verified main sig')
                       
    return crystallized_state, o

def mk_genesis_state_and_block(pubkeys):
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
        crosslink_records=[CrosslinkRecord(hash=b'\x00'*32, epoch=0) for i in range(SHARD_COUNT)],
        total_deposits=DEFAULT_BALANCE*len(pubkeys))
    a = ActiveState(height=1,
        randao=b'\x45'*32,
        ffg_voter_bitmask=bytearray((len(c.active_validators) + 7) // 8),
        balance_deltas=[],
        checkpoints=[],
        total_skip_count=0)
    b = Block(parent_hash=b'\x00'*32,
        skip_count=0,
        randao_reveal=b'\x00'*32,
        attestation_bitmask=b'',
        attestation_aggregate_sig=[0,0],
        shard_aggregate_votes=[],
        main_chain_ref=b'\x00'*32,
        state_hash=blake(serialize(c))+blake(serialize(a)),
        sig=[0,0]
    )
    return c, a, b
