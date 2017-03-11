# Information about validators
validators: {
    # Amount of wei the validator holds
    deposit: wei_value,
    # The dynasty the validator is joining
    dynasty_start: num,
    # The dynasty the validator is leaving
    dynasty_end: num,
    # The timestamp at which the validator can withdraw
    withdrawal_time: timestamp,
    # The address which the validator's signatures must verify to (to be later replaced with validation code)
    addr: address,
    # Addess to withdraw to
    withdrawal_addr: address,
    # The max epoch at which the validator prepared
    max_prepared: num,
    # The max epoch at which the validator committed
    max_committed: num
}[num]

# The current dynasty (validator set changes between dynasties)
dynasty: num
# Amount of wei added to the total deposits in the next dynasty
next_dynasty_wei_delta: wei_value
# Amount of wei added to the total deposits in the dynasty after that
second_next_dynasty_wei_delta: wei_value

# Total deposits during this dynasty
total_deposits: wei_value[num]

# Information for use in processing cryptoeconomic commitments
consensus_messages: {
    # How many prepares are there for this hash (hash of message hash + view source) from the current dynasty
    prepares: wei_value[bytes32],
    # From the previous dynasty
    prev_dyn_prepares: wei_value[bytes32],
    # Is a commit on the given hash justified?
    justified: bool[bytes32],
    # How many commits are there for this hash
    commits: wei_value[bytes32],
    # And from the previous dynasty
    prev_dyn_commits: wei_value[bytes32],
    # Was the block committed?
    committed: bool
}[num] # index: epoch

# ancestry[x][y] = k > 0: x is a kth generation ancestor of y
ancestry: num[bytes32][bytes32]

# Number of validators
nextValidatorIndex: num

# Constant that guides the size of validator rewards
interest_rate: decimal(1 / sec)

# Time between blocks
block_time: timedelta

# Length of an epoch in blocks
epoch_length: num

# Withdrawal delay
withdrawal_delay: timedelta

# Delay after which a message can be slashed due to absence of justification
insufficiency_slash_delay: timedelta

# Current epoch
current_epoch: num

# Can withdraw destroyed deposits
owner: address

# Total deposits destroyed
total_destroyed: wei_value

# RLP decoder address
rlp_decoder: address

def __init__():
    # Set Casper parameters
    self.interest_rate = 0.000001
    self.block_time = 7
    self.epoch_length = 256
    # Only ~11.5 days, for testing purposes
    self.withdrawal_delay = 1000000
    # Temporary backdoor for testing purposes (to allow recovering destroyed deposits)
    self.owner = 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6
    # Add an initial validator
    self.validators[0] = {
        deposit: as_wei(3, finney),
        dynasty_start: 0,
        dynasty_end: 1000000000000000000000000000000,
        withdrawal_time: 1000000000000000000000000000000,
        addr: 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6,
        withdrawal_addr: 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6,
        max_prepared: 0,
        max_committed: 0
    }
    # Initialize the epoch counter
    self.current_epoch = block.number / self.epoch_length
    # Set the RLP decoder address
    self.rlp_decoder = 0x38920146f10f3956fc09970beededcb2d9638711

# Called at the start of any epoch
def initialize_epoch(epoch: num):
    computed_current_epoch = block.number / self.epoch_length
    if epoch <= computed_current_epoch and epoch == self.current_epoch + 1:
        self.current_epoch = epoch
        if self.consensus_messages[epoch - 1].committed:
            self.dynasty += 1
            self.total_deposits[self.dynasty] = self.total_deposits[self.dynasty - 1] + self.next_dynasty_wei_delta
            self.next_dynasty_wei_delta = self.second_next_dynasty_wei_delta
            self.second_next_dynasty_wei_delta = 0

# Send a deposit to join the validator set
def deposit(validation_addr: address, withdrawal_addr: address):
    assert self.current_epoch == block.number / self.epoch_length
    self.validators[self.nextValidatorIndex] = {
        deposit: msg.value,
        dynasty_start: self.dynasty + 2,
        dynasty_end: 1000000000000000000000000000000,
        withdrawal_time: 1000000000000000000000000000000,
        addr: validation_addr,
        withdrawal_addr: withdrawal_addr,
        max_prepared: 0,
        max_committed: 0,
    }
    self.nextValidatorIndex += 1
    self.insufficiency_slash_delay = self.withdrawal_delay / 2
    self.second_next_dynasty_wei_delta += msg.value

# Exit the validator set, and start the withdrawal procedure
def start_withdrawal(index: num, sig: bytes <= 96):
    assert self.current_epoch == block.number / self.epoch_length
    # Signature check
    assert len(sig) == 96
    assert ecrecover(sha3("withdraw"),
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that we haven't already withdrawn
    assert self.validators[index].dynasty_end >= self.dynasty + 2
    # Set the end dynasty
    self.validators[index].dynasty_end = self.dynasty + 2
    self.second_next_dynasty_wei_delta -= msg.value
    # Set the withdrawal date
    self.validators[index].withdrawal_time = block.timestamp + self.withdrawal_delay

# Withdraw deposited ether
def withdraw(index: num):
    # Check that we can withdraw
    assert block.timestamp >= self.validators[index].withdrawal_time
    # Withdraw
    send(self.validators[index].withdrawal_addr, self.validators[index].deposit)
    self.validators[index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

# Process a prepare message
def prepare(index: num, epoch: num, hash: bytes32, ancestry_hash: bytes32,
            epoch_source: num, source_ancestry_hash: bytes32, sig: bytes <= 96):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, self.prepare_msg, gas=50000, outsize=32), 0)
    values = raw_call(self.rlp_decoder, self.prepare_msg, gas=50000, outsize=2048)
    # Extract parameters
    epoch = extract32(values, extract32(value, 0, type=num128), type=num128)
    hash = extract32(values, extract32(value, 32, type=num128), type=bytes32)
    ancestry_hash = extract32(values, extract32(value, 64, type=num128), type=bytes32)
    source_epoch = extract32(values, extract32(value, 96, type=num128), type=num128)
    source_ancestry_hash = extract32(values, extract32(value, 128, type=num128), type=bytes32)
    sig = slice(values, start=extract32(value, 160, type=num128), end=extract32(value, 192, type=num128))
    # Assert that there are only 6 elements
    assert as_num128(extract32(value, 0)) == 224
    # For now, the sig is a simple ECDSA sig
    assert len(sig) == 96
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    # Check that this validator was active in either the previous dynasty or the current one
    ds = self.validators[index].dynasty_start
    de = self.validators[index].dynasty_end
    dc = self.dynasty
    in_current_dynasty = (ds <= dc) and (dc < de)
    in_prev_dynasty = (ds <= (dc - 1)) and ((dc - 1) < de)
    assert in_current_dynasty or in_prev_dynasty
    # Check that we have not yet prepared for this epoch
    assert self.validators[index].max_prepared == epoch - 1
    # Pay the reward if the blockhash is correct
    if True: #~blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[index].deposit * self.interest_rate * self.block_time)
        self.validators[index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't prepare for this epoch again
    self.validators[index].max_prepared = epoch
    # Record that this prepare took place
    new_ancestry_hash = sha3(concat(hash, ancestry_hash))
    if in_current_dynasty:
        self.consensus_messages[epoch].prepares[sighash] += self.validators[index].deposit
    if in_prev_dynasty:
        self.consensus_messages[epoch].prev_dyn_prepares[sighash] += self.validators[index].deposit
    # If enough prepares with the same epoch_source and hash are made,
    # then the hash value is justified for commitment
    if (self.consensus_messages[epoch].prepares[sighash] >= self.total_deposits[self.dynasty] * 2 / 3 and \
            self.consensus_messages[epoch].prev_dyn_prepares[sighash] >= self.total_deposits[self.dynasty] * 2 / 3) and \
            not self.consensus_messages[epoch].justified[new_ancestry_hash]:
        self.consensus_messages[epoch].justified[new_ancestry_hash] = True
    # Add a parent-child relation between ancestry hashes to the ancestry table
    self.ancestry[ancestry_hash][new_ancestry_hash] = 1

# Process a commit message
def commit(index: num, epoch: num, hash: bytes32, sig: bytes <= 96):
    # Signature check
    sighash = sha3(concat("commit", as_bytes32(epoch), hash))
    assert len(sig) == 96
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    # Check that this validator was active in either the previous dynasty or the current one
    ds = self.validators[index].dynasty_start
    de = self.validators[index].dynasty_end
    dc = self.dynasty
    in_current_dynasty = (ds <= dc) and (dc < de)
    in_prev_dynasty = (ds <= (dc - 1)) and ((dc - 1) < de)
    assert in_current_dynasty or in_prev_dynasty
    # Check that we have not yet committed for this epoch
    assert self.validators[index].max_committed == epoch - 1
    # Pay the reward if the blockhash is correct
    if True: #~blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[index].deposit * self.interest_rate * self.block_time)
        self.validators[index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't commit for this epoch again
    self.validators[index].max_committed = epoch
    # Record that this commit took place
    if in_current_dynasty:
        self.consensus_messages[epoch].commits[hash] += self.validators[index].deposit
    if in_prev_dynasty:
        self.consensus_messages[epoch].prev_dyn_commits[hash] += self.validators[index].deposit
    # Record if sufficient commits have been made for the block to be finalized
    if self.consensus_messages[epoch].commits[hash] >= self.total_deposits[self.dynasty] * 2 / 3 and \
            not self.consensus_messages[epoch].committed:
        self.consensus_messages[epoch].committed = True

# Cannot make two versions of the same message
def intra_epoch_equivocation_slash(index: num, epoch: num, msgtype: bytes <= 16,
                                   args1: bytes <= 250, args2: bytes <= 250, sig1: bytes <= 96, sig2: bytes<= 96):
    # Signature check
    sighash1 = sha3(concat(msgtype, as_bytes32(epoch), args1))
    sighash2 = sha3(concat(msgtype, as_bytes32(epoch), args2))
    assert ecrecover(sighash1,
                     as_num256(extract32(sig1, 0)),
                     as_num256(extract32(sig1, 32)),
                     as_num256(extract32(sig1, 64))) == self.validators[index].addr
    assert ecrecover(sighash2,
                     as_num256(extract32(sig2, 0)),
                     as_num256(extract32(sig2, 32)),
                     as_num256(extract32(sig2, 64))) == self.validators[index].addr
    # Check that they're not the same message
    assert sighash1 != sighash2
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= (validator_deposit - validator_deposit / 25)
    self.validators[index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

def prepare_commit_inconsistency_slash(index: num, prepare_epoch: num, prepare_hash: bytes32,
                                       prepare_source_epoch: num, prepare_source_ancestry_hash: bytes32, sig1: bytes <= 96,
                                       commit_epoch: num, commit_hash: bytes32, sig2: bytes <= 96):
    # Signature check
    sighash1 = sha3(concat("prepare", as_bytes32(prepare_epoch), prepare_hash,
                           as_bytes32(prepare_source_epoch), prepare_source_ancestry_hash))
    sighash2 = sha3(concat("commit", as_bytes32(commit_epoch), commit_hash))
    assert ecrecover(sighash1,
                     as_num256(extract32(sig1, 0)),
                     as_num256(extract32(sig1, 32)),
                     as_num256(extract32(sig1, 64))) == self.validators[index].addr
    assert ecrecover(sighash2,
                     as_num256(extract32(sig2, 0)),
                     as_num256(extract32(sig2, 32)),
                     as_num256(extract32(sig2, 64))) == self.validators[index].addr
    # Check that they're not the same message
    assert sighash1 != sighash2
    # Check that the prepare refers to something older than the commit
    assert prepare_source_epoch < commit_epoch
    # Check that the prepare is newer than the commit
    assert commit_epoch < prepare_epoch
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

def commit_non_justification_slash(index: num, epoch: num, hash: bytes32, sig: bytes <= 96):
    # Signature check
    sighash = sha3(concat("commit", as_bytes32(epoch), hash))
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that the commit is old enough
    assert self.current_epoch == block.number / self.epoch_length
    assert (epoch - self.current_epoch) * self.block_time > self.insufficiency_slash_delay
    assert not self.consensus_messages[epoch].justified[hash]
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

# Fill in the table for which hash is what-degree ancestor of which other hash
def derive_ancestry(top: bytes32, middle: bytes32, bottom: bytes32):
    assert self.ancestry[middle][top]
    assert self.ancestry[bottom][middle]
    self.ancestry[bottom][top] = self.ancestry[bottom][middle] + self.ancestry[middle][top]

def prepare_non_justification_slash(index: num, epoch: num, hash: bytes32, epoch_source: num,
                                    source_ancestry_hash: bytes32, sig: bytes <= 96):
    # Signature check
    sighash = sha3(concat("viewchange", as_bytes32(epoch), hash, as_bytes32(epoch_source), source_ancestry_hash))
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that the view change is old enough
    assert self.current_epoch == block.number / self.epoch_length
    assert (epoch - self.current_epoch) * self.block_time > self.insufficiency_slash_delay
    # Check that the source ancestry hash had enough prepares
    assert not self.consensus_messages[epoch_source].justified[source_ancestry_hash]
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

# Temporary backdoor for testing purposes (to allow recovering destroyed deposits)
def owner_withdraw():
    send(self.owner, self.total_destroyed)
    self.total_destroyed = 0

# Change backdoor address (set to zero to remove entirely)
def change_owner(new_owner: address):
    if self.owner == msg.sender:
        self.owner = new_owner
