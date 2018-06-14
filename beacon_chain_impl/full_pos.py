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
        'shard_block_hash': 'hash32',
        'signer_bitmask': 'bytes',
        'aggregate_sig': ['int256']
    }
    defaults = {
        'shard_id': 0,
        'shard_block_hash': b'\x00'*32,
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
    fields = {
        # The validator's public key
        'pubkey': 'int256',
        # What shard the validator's balance will be sent to after withdrawal
        'return_shard': 'int16',
        # And what address
        'return_address': 'address',
        # The validator's current RANDAO beacon commitment
        'randao_commitment': 'hash32',
        # Current balance
        'balance': 'int64',
        # Dynasty where the validator can (be inducted | be removed | withdraw)
        'switch_dynasty': 'int64'
    }
    defaults = {}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

class PartialCrosslinkRecord():

    fields = {
        # What shard is the crosslink being made for
        'shard_id': 'int16',
        # Hash of the block
        'shard_block_hash': 'hash32',
        # Which of the eligible voters are voting for it (as a bitmask)
        'voter_bitmask': 'bytes'
    }
    defaults = {}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults, k
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))


class ActiveState():

    fields = {
        # Block height
        'height': 'int64',
        # Global RANDAO beacon state
        'randao': 'hash32',
        # Which validators have made FFG votes this epoch (as a bitmask)
        'ffg_voter_bitmask': 'bytes',
        # Deltas to validator balances (to be processed at end of epoch)
        'balance_deltas': ['int48'],
        # Storing data about crosslinks-in-progress attempted in this epoch
        'partial_crosslinks': [PartialCrosslinkRecord],
        # Total number of skips (used to determine minimum timestamp)
        'total_skip_count': 'int64'
    }
    defaults = {'height': 0, 'randao': b'\x00'*32,
        'ffg_voter_bitmask': b'', 'balance_deltas': [],
        'partial_crosslinks': [], 'total_skip_count': 0}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))

class CrosslinkRecord():
    fields = {
        # What epoch the crosslink was submitted in
        'epoch': 'int64',
        # The block hash
        'hash': 'hash32'
    }
    defaults = {'epoch': 0, 'hash': b'\x00'*32}

    def __init__(self, **kwargs):
        for k in self.fields.keys():
            assert k in kwargs or k in self.defaults
            setattr(self, k, kwargs.get(k, self.defaults.get(k)))
                                        
class CrystallizedState():
    fields = {
        # List of active validators
        'active_validators': [ValidatorRecord],
        # List of joined but not yet inducted validators
        'queued_validators': [ValidatorRecord],
        # List of removed validators pending withdrawal
        'exited_validators': [ValidatorRecord],
        # The permutation of validators used to determine who cross-links
        # what shard in this epoch
        'current_shuffling': ['int24'],
        # The current epoch
        'current_epoch': 'int64',
        # The last justified epoch
        'last_justified_epoch': 'int64',
        # The last finalized epoch
        'last_finalized_epoch': 'int64',
        # The current dynasty
        'dynasty': 'int64',
        # The next shard that assignment for cross-linking will start from
        'next_shard': 'int16',
        # The current FFG checkpoint
        'current_checkpoint': 'hash32',
        # Records about the most recent crosslink for each shard
        'crosslink_records': [CrosslinkRecord],
        # Total balance of deposits
        'total_deposits': 'int256'
    }
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

def get_crosslink_aggvote_msg(shard_id, shard_block_hash, crystallized_state):
    return shard_id.to_bytes(2, 'big') + \
        shard_block_hash + \
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

# Get rewards and vote data
def process_ffg_deposits(crystallized_state, ffg_voter_bitmask):
    total_validators = len(crystallized_state.active_validators)
    finality_distance = crystallized_state.current_epoch - crystallized_state.last_finalized_epoch
    online_reward = 6 if finality_distance <= 2 else 0
    offline_penalty = 3 * finality_distance
    total_vote_count = 0
    total_vote_deposits = 0
    deltas = [0] * total_validators
    for i in range(total_validators):
        if ffg_voter_bitmask[i // 8] & (128 >> (i % 8)):
            total_vote_deposits += crystallized_state.active_validators[i].balance
            deltas[i] += online_reward
            total_vote_count += 1
        else:
            deltas[i] -= offline_penalty
    print('Total voted: %d of %d validators (%.2f%%), %d of %d deposits (%.2f%%)' %
          (total_vote_count, total_validators, total_vote_count * 100 / total_validators,
           total_vote_deposits, crystallized_state.total_deposits, total_vote_deposits * 100 / crystallized_state.total_deposits))
    print('FFG online reward: %d, offline penalty: %d' % (online_reward, offline_penalty))
    print('Total deposit change from FFG: %d' % sum(deltas))
    # Check if we need to justify and finalize
    justify = total_vote_deposits * 3 >= crystallized_state.total_deposits * 2
    finalize = False
    if justify:
        print('Justifying last epoch')
        if crystallized_state.last_justified_epoch == crystallized_state.current_epoch - 1:
            finalize = True
            print('Finalizing last epoch')
    return deltas, total_vote_count, total_vote_deposits, justify, finalize

# Process rewards from crosslinks
def process_crosslinks(crystallized_state, crosslinks):
    # Find the most popular crosslink in each shard
    main_crosslink = {}
    for c in crosslinks:
        vote_count = 0
        mask =  bytearray(c.voter_bitmask)
        for byte in mask:
            for j in range(8):
                vote_count += (byte >> j) % 2
        if vote_count > main_crosslink.get(c.shard_id, (b'', 0, b''))[1]:
            main_crosslink[c.shard_id] = (c.shard_block_hash, vote_count, mask)
    # Adjust crosslinks
    new_crosslink_records = [x for x in crystallized_state.crosslink_records]
    deltas = [0] * len(crystallized_state.active_validators)
    # Process shard by shard...
    for shard in range(SHARD_COUNT):
        indices = get_shard_attesters(crystallized_state, shard)
        # Get info about the dominant crosslink for this shard
        h, votes, mask = main_crosslink.get(shard, (b'', 0, bytearray((len(indices)+7)//8)))
        # Calculate rewards for participants and penalties for non-participants
        crosslink_distance = crystallized_state.current_epoch - crystallized_state.crosslink_records[shard].epoch
        online_reward = 3 if crosslink_distance <= 2 else 0
        offline_penalty = crosslink_distance * 2
        # Go through participants and evaluate rewards/penalties
        for i, index in enumerate(indices):
            if mask[i//8] & (1 << (i % 8)):
                deltas[i] += online_reward
            else:
                deltas[i] -= offline_penalty
        print('Shard %d: most recent crosslink %d, reward: (%d, %d), votes: %d of %d (%.2f%%)'
              % (shard, crystallized_state.crosslink_records[shard].epoch, online_reward, -offline_penalty,
                 votes, len(indices), votes * 100 / len(indices)))
        # New crosslink
        if votes * 3 >= len(indices) * 2:
            new_crosslink_records[shard] = CrosslinkRecord(hash=h, epoch=crystallized_state.current_epoch)
            print('New crosslink %s' % hex(int.from_bytes(h, 'big')))
    print('Total deposit change from crosslinks: %d' % sum(deltas))
    return deltas, new_crosslink_records

def process_balance_deltas(crystallized_state, balance_deltas):
    deltas = [0] * len(crystallized_state.active_validators)
    for i in balance_deltas:
        if i % 16777216 < 8388608:
            deltas[i >> 24] += i & 16777215
        else:
            deltas[i >> 24] += (i & 16777215) - 16777216
    print('Total deposit change from deltas: %d' % sum(deltas))
    return deltas

def get_incremented_validator_sets(crystallized_state, new_active_validators):
    new_active_validators = [v for v in new_active_validators]
    new_exited_validators = [v for v in crystallized_state.exited_validators]
    i = 0
    while i < len(new_active_validators):
        if new_active_validators[i].balance <= DEFAULT_BALANCE // 2:
            new_exited_validators.append(new_active_validators.pop(i))
        elif new_active_validators[i].switch_dynasty == crystallized_state.dynasty + 1:
            new_exited_validators.append(new_active_validators.pop(i))
        else:
            i += 1
    induct = min(len(crystallized_state.queued_validators), len(crystallized_state.active_validators) // 30 + 1)
    for i in range(induct):
        if crystallized_state.queued_validators[i].switch_dynasty > crystallized_state.dynasty + 1:
            induct = i
            break
        new_active_validators.append(crystallized_state.queued_validators[i])
    new_queued_validators = crystallized_state.queued_validators[induct:]
    return new_queued_validators, new_active_validators, new_exited_validators

def process_attestations(validator_set, attestation_indices, attestation_bitmask, msg, aggregate_sig):
    # Verify the attestations of the parent
    pubs = []
    balance_deltas = []
    assert len(attestation_bitmask) == (len(attestation_indices) + 7) // 8
    for i, index in enumerate(attestation_indices):
        if attestation_bitmask[i//8] & (128>>(i%8)):
            pubs.append(validator_set[index].pubkey)
            balance_deltas.append((index << 24) + 1)
    assert len(balance_deltas) <= 128
    assert verify(msg, aggregate_pubs(pubs), aggregate_sig)
    print('Verified aggregate sig')
    return balance_deltas


def update_ffg_and_crosslink_progress(crystallized_state, crosslinks, ffg_voter_bitmask, votes):
    # Verify the attestations of crosslink hashes
    crosslink_votes = {vote.shard_block_hash + vote.shard_id.to_bytes(2, 'big'):
                        vote.voter_bitmask for vote in crosslinks} 
    new_ffg_bitmask = bytearray(ffg_voter_bitmask)
    total_voters = 0
    for vote in votes:
        attestation = get_crosslink_aggvote_msg(vote.shard_id, vote.shard_block_hash, crystallized_state)
        indices = get_shard_attesters(crystallized_state, vote.shard_id)
        votekey = vote.shard_block_hash + vote.shard_id.to_bytes(2, 'big')
        if votekey not in crosslink_votes:
            crosslink_votes[votekey] = bytearray((len(indices) + 7) // 8)
        bitmask = crosslink_votes[votekey]
        pubs = []
        for i, index in enumerate(indices):
            if vote.signer_bitmask[i//8] & (128>>(i%8)):
                pubs.append(crystallized_state.active_validators[index].pubkey)
                if new_ffg_bitmask[index//8] & (128>>(index%8)) == 0:
                    new_ffg_bitmask[index//8] ^= 128>>(index%8)
                    bitmask[i//8] ^= 128>>(i%8)
                    total_voters += 1
        assert verify(attestation, aggregate_pubs(pubs), vote.aggregate_sig)
        crosslink_votes[votekey] = bitmask
        print('Verified aggregate vote')
    new_crosslinks = [PartialCrosslinkRecord(shard_id=int.from_bytes(h[32:], 'big'),
                      shard_block_hash=h[:32], voter_bitmask=crosslink_votes[h])
                      for h in sorted(crosslink_votes.keys())]
    return new_crosslinks, new_ffg_bitmask, total_voters

def compute_state_transition(parent_state, parent_block, block, verify_sig=True):
    crystallized_state, active_state = parent_state
    # Initialize a new epoch if needed
    if active_state.height % SHARD_COUNT == 0:
        print('Processing epoch transition')
        # Process rewards from FFG/crosslink votes
        new_validator_records = deepcopy(crystallized_state.active_validators)
        # Who voted in the last epoch
        ffg_voter_bitmask = bytearray(active_state.ffg_voter_bitmask)
        # Balance changes, and total vote counts for FFG
        deltas1, total_vote_count, total_vote_deposits, justify, finalize = \
            process_ffg_deposits(crystallized_state, ffg_voter_bitmask)
        # Balance changes, and total vote counts for crosslinks
        deltas2, new_crosslink_records = process_crosslinks(crystallized_state, active_state.partial_crosslinks)
        # Process other balance deltas
        deltas3 = process_balance_deltas(crystallized_state, active_state.balance_deltas)
        for i, v in enumerate(new_validator_records):
            v.balance += deltas1[i] + deltas2[i] + deltas3[i]
        total_deposits = crystallized_state.total_deposits + sum(deltas1 + deltas2 + deltas3)
        print('New total deposits: %d' % total_deposits)

        if finalize:
            new_queued_validators, new_active_validators, new_exited_validators = \
                get_incremented_validator_sets(crystallized_state, new_validator_records)
        else:
            new_queued_validators, new_active_validators, new_exited_validators = \
                crystallized_state.queued_validators, crystallized_state.active_validators, crystallized_state.exited_validators

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
                                   partial_crosslinks=[],
                                   total_skip_count=active_state.total_skip_count)
    # Process the block-by-block stuff

    # Determine who the attesters and the main signer are
    attestation_indices, main_signer = \
        get_attesters_and_signer(crystallized_state, active_state, block.skip_count)

    # Verify attestations
    balance_deltas = process_attestations(crystallized_state.active_validators,
                                          attestation_indices,
                                          block.attestation_bitmask,
                                          serialize(parent_block),
                                          block.attestation_aggregate_sig)
    # Reward main signer
    balance_deltas.append((main_signer << 24) + len(balance_deltas))

    # Verify main signature
    if verify_sig:
        assert block.verify(crystallized_state.active_validators[main_signer].pubkey)
        print('Verified main sig')

    # Update crosslink records
    new_crosslink_records, new_ffg_bitmask, voters = \
        update_ffg_and_crosslink_progress(crystallized_state, active_state.partial_crosslinks,
                                          active_state.ffg_voter_bitmask, block.shard_aggregate_votes)
    balance_deltas.append((main_signer << 24) + voters)
    
    o =  ActiveState(height=active_state.height + 1,
                     randao=(int.from_bytes(active_state.randao, 'big') ^ 
                             int.from_bytes(block.randao_reveal, 'big')).to_bytes(32, 'big'),
                     total_skip_count=active_state.total_skip_count + block.skip_count,
                     partial_crosslinks=new_crosslink_records,
                     ffg_voter_bitmask=new_ffg_bitmask,
                     balance_deltas=active_state.balance_deltas + balance_deltas)
                       
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
        partial_crosslinks=[],
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
