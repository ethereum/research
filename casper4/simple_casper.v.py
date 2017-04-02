# Information about validators
validators: public({
    # Amount of wei the validator holds
    deposit: wei_value,
    # The dynasty the validator is joining
    dynasty_start: num,
    # The dynasty the validator is leaving
    dynasty_end: num,
    # The timestamp at which the validator can withdraw
    withdrawal_epoch: num,
    # The address which the validator's signatures must verify to (to be later replaced with validation code)
    addr: address,
    # Addess to withdraw to
    withdrawal_addr: address,
    # The max epoch at which the validator prepared
    max_prepared: num,
    # The max epoch at which the validator committed
    max_committed: num
}[num])

# The current dynasty (validator set changes between dynasties)
dynasty: public(num)
# Amount of wei added to the total deposits in the next dynasty
next_dynasty_wei_delta: wei_value
# Amount of wei added to the total deposits in the dynasty after that
second_next_dynasty_wei_delta: wei_value

# Total deposits during this dynasty
total_deposits: public(wei_value[num])

# Mapping of dynasty to start epoch of that dynasty
dynasty_start_epoch: public(num[num])

# Information for use in processing cryptoeconomic commitments
consensus_messages: public({
    # How many prepares are there for this hash (hash of message hash + view source) from the current dynasty
    prepares: wei_value[bytes32],
    # From the previous dynasty
    prev_dyn_prepares: wei_value[bytes32],
    # Is a prepare referencing the given ancestry hash justified?
    ancestry_hash_justified: bool[bytes32],
    # Is a commit on the given hash justified?
    hash_justified: bool[bytes32],
    # How many commits are there for this hash
    commits: wei_value[bytes32],
    # And from the previous dynasty
    prev_dyn_commits: wei_value[bytes32],
    # Was the block committed?
    committed: bool,
    # Value used to calculate the per-epoch fee that validators should be charged
    deposit_scale_factor: decimal
}[num]) # index: epoch

# ancestry[x][y] = k > 0: x is a kth generation ancestor of y
ancestry: public(num[bytes32][bytes32])

# Number of validators
nextValidatorIndex: public(num)

# Time between blocks
block_time: timedelta

# Length of an epoch in blocks
epoch_length: num

# Withdrawal delay
withdrawal_delay: timedelta

# Delay after which a message can be slashed due to absence of justification
insufficiency_slash_delay: timedelta

# Current epoch
current_epoch: public(num)

# Can withdraw destroyed deposits
owner: address

# Total deposits destroyed
total_destroyed: wei_value

# Sighash calculator library address
sighasher: address


# Reward for preparing or committing, as fraction of deposit size
reward_factor: public(decimal)

# Desired total ether given out assuming 1M ETH deposited
reward_at_1m_eth: decimal

# Have I already been initialized?
initialized: bool

def initiate():
    assert not self.initialized
    self.initialized = True
    # Set Casper parameters
    self.block_time = 14
    self.epoch_length = 100
    # Only ~11.5 days, for testing purposes
    self.withdrawal_delay = 1000000
    # Only ~1 day, for testing purposes
    self.insufficiency_slash_delay = 86400
    # Temporary backdoor for testing purposes (to allow recovering destroyed deposits)
    self.owner = 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6
    # Add an initial validator
    self.validators[0] = {
        deposit: as_wei_value(3, ether),
        dynasty_start: 0,
        dynasty_end: 1000000000000000000000000000000,
        withdrawal_epoch: 1000000000000000000000000000000,
        addr: 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6,
        withdrawal_addr: 0x1db3439a222c519ab44bb1144fc28167b4fa6ee6,
        max_prepared: 0,
        max_committed: 0
    }
    self.nextValidatorIndex = 1
    # Initialize the epoch counter
    self.current_epoch = block.number / self.epoch_length
    # Set the sighash calculator address
    self.sighasher = 0x476c2ca9a7f3b16feca86512276271faf63b6a24
    # Set an initial root of the epoch hash chain
    self.consensus_messages[0].ancestry_hash_justified[0x0000000000000000000000000000000000000000000000000000000000000000] = True
    # Set initial total deposit counter
    self.total_deposits[0] = as_wei_value(3, ether)
    # Set deposit scale factor
    self.consensus_messages[0].deposit_scale_factor = 1000000000000000000.0
    # Total ETH given out assuming 1m ETH deposits
    self.reward_at_1m_eth = 12.5

# Called at the start of any epoch
def initialize_epoch(epoch: num):
    # Check that the epoch actually has started
    computed_current_epoch = block.number / self.epoch_length
    assert epoch <= computed_current_epoch and epoch == self.current_epoch + 1
    # Set the epoch number
    self.current_epoch = epoch
    # Increment the dynasty
    if self.consensus_messages[epoch - 1].committed:
        self.dynasty += 1
        self.total_deposits[self.dynasty] = self.total_deposits[self.dynasty - 1] + self.next_dynasty_wei_delta
        self.next_dynasty_wei_delta = self.second_next_dynasty_wei_delta
        self.second_next_dynasty_wei_delta = 0
        self.dynasty_start_epoch[self.dynasty] = epoch
    # Compute square root factor
    ether_deposited_as_number = self.total_deposits[self.dynasty] / as_wei_value(1, ether)
    sqrt = ether_deposited_as_number / 2.0
    for i in range(20):
        sqrt = (sqrt + (ether_deposited_as_number / sqrt)) / 2
    # Reward factor is the reward given for preparing or committing as a
    # fraction of that validator's deposit size
    base_coeff = 1.0 / sqrt * (self.reward_at_1m_eth / 1000)
    # Rules:
    # * You are penalized 2x per epoch
    # * If you prepare, you get 1.5x, and if you commit you get another 1.5x
    # Hence, assuming 100% performance, your reward per epoch is x
    self.reward_factor = 1.5 * base_coeff
    self.consensus_messages[epoch].deposit_scale_factor = self.consensus_messages[epoch - 1].deposit_scale_factor * (1 - 2 * base_coeff)

# Send a deposit to join the validator set
def deposit(validation_addr: address, withdrawal_addr: address):
    assert self.current_epoch == block.number / self.epoch_length
    self.validators[self.nextValidatorIndex] = {
        deposit: msg.value,
        dynasty_start: self.dynasty + 2,
        dynasty_end: 1000000000000000000000000000000,
        withdrawal_epoch: 1000000000000000000000000000000,
        addr: validation_addr,
        withdrawal_addr: withdrawal_addr,
        max_prepared: 0,
        max_committed: 0,
    }
    self.nextValidatorIndex += 1
    self.second_next_dynasty_wei_delta += msg.value

# Log in or log out from the validator set. A logged out validator can log
# back in later, if they do not log in for an entire withdrawal period,
# they can get their money out
def flick_status(validator_index: num, logout_msg: bytes <= 1024):
    assert self.current_epoch == block.number / self.epoch_length
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, logout_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(logout_msg, [num, bool, bytes])
    epoch = values[0]
    login_flag = values[1]
    sig = values[2]
    assert self.current_epoch == epoch
    # Signature check
    assert len(sig) == 96
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[validator_index].addr
    # Logging in
    if login_flag:
        # Check that we are logged out
        assert self.validators[validator_index].dynasty_end < self.dynasty
        # Apply the per-epoch deposit penalty
        prev_login_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_start]
        prev_logout_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_end + 1]
        self.validators[validator_index].deposit = \
            floor(self.validators[validator_index].deposit *
                    (self.consensus_messages[prev_logout_epoch].deposit_scale_factor /
                     self.consensus_messages[prev_login_epoch].deposit_scale_factor))
        # Log back in
        self.validators[validator_index].dynasty_start = self.dynasty + 2
        self.validators[validator_index].dynasty_end = 1000000000000000000000000000000
        self.second_next_dynasty_wei_delta += self.validators[validator_index].deposit
    # Logging out
    else:
        # Check that we haven't already withdrawn
        assert self.validators[validator_index].dynasty_end >= self.dynasty + 2
        # Set the end dynasty
        self.validators[validator_index].dynasty_end = self.dynasty + 2
        self.second_next_dynasty_wei_delta -= self.validators[validator_index].deposit
        # Set the withdrawal date
        self.validators[validator_index].withdrawal_epoch = self.current_epoch + self.withdrawal_delay / self.block_time / self.epoch_length

# Withdraw deposited ether
def withdraw(validator_index: num):
    # Check that we can withdraw
    assert self.current_epoch >= self.validators[validator_index].withdrawal_epoch
    # Apply the per-epoch deposit penalty
    prev_login_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_start]
    prev_logout_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_end + 1]
    self.validators[validator_index].deposit = \
        floor(self.validators[validator_index].deposit *
                (self.consensus_messages[prev_logout_epoch].deposit_scale_factor /
                 self.consensus_messages[prev_login_epoch].deposit_scale_factor))
    # Withdraw
    send(self.validators[validator_index].withdrawal_addr, self.validators[validator_index].deposit)
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_epoch: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

# Process a prepare message
def prepare(validator_index: num, prepare_msg: bytes <= 1024):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(prepare_msg, [num, bytes32, bytes32, num, bytes32, bytes])
    epoch = values[0]
    hash = values[1]
    ancestry_hash = values[2]
    source_epoch = values[3]
    source_ancestry_hash = values[4]
    sig = values[5]
    # For now, the sig is a simple ECDSA sig
    # Check the signature
    assert len(sig) == 96
    assert ecrecover(sighash,
                     extract32(sig, 0, type=num256),
                     extract32(sig, 32, type=num256),
                     extract32(sig, 64, type=num256)) == self.validators[validator_index].addr
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    # Check that we are at least (epoch length / 4) blocks into the epoch
    # assert block.number % self.epoch_length >= self.epoch_length / 4
    # Check that this validator was active in either the previous dynasty or the current one
    ds = self.validators[validator_index].dynasty_start
    de = self.validators[validator_index].dynasty_end
    dc = self.dynasty
    in_current_dynasty = (ds <= dc) and (dc < de)
    in_prev_dynasty = (ds <= (dc - 1)) and ((dc - 1) < de)
    assert in_current_dynasty or in_prev_dynasty
    # Check that the prepare is on top of a justified prepare
    assert self.consensus_messages[source_epoch].ancestry_hash_justified[source_ancestry_hash]
    # Check that we have not yet prepared for this epoch
    #assert self.validators[validator_index].max_prepared == epoch - 1
    # Pay the reward if the blockhash is correct
    if True:  #if blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[validator_index].deposit * self.reward_factor)
        self.validators[validator_index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't prepare for this epoch again
    self.validators[validator_index].max_prepared = epoch
    # Record that this prepare took place
    new_ancestry_hash = sha3(concat(hash, ancestry_hash))
    if in_current_dynasty:
        self.consensus_messages[epoch].prepares[sighash] += self.validators[validator_index].deposit
    if in_prev_dynasty:
        self.consensus_messages[epoch].prev_dyn_prepares[sighash] += self.validators[validator_index].deposit
    # If enough prepares with the same epoch_source and hash are made,
    # then the hash value is justified for commitment
    if (self.consensus_messages[epoch].prepares[sighash] >= self.total_deposits[self.dynasty] * 2 / 3 and \
            self.consensus_messages[epoch].prev_dyn_prepares[sighash] >= self.total_deposits[self.dynasty - 1] * 2 / 3) and \
            not self.consensus_messages[epoch].ancestry_hash_justified[new_ancestry_hash]:
        self.consensus_messages[epoch].ancestry_hash_justified[new_ancestry_hash] = True
        self.consensus_messages[epoch].hash_justified[hash] = True
    # Add a parent-child relation between ancestry hashes to the ancestry table
    self.ancestry[ancestry_hash][new_ancestry_hash] = 1

# Process a commit message
def commit(validator_index: num, commit_msg: bytes <= 1024):
    sighash = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(commit_msg, [num, bytes32, bytes])
    epoch = values[0]
    hash = values[1]
    sig = values[2]
    # Check the signature
    assert len(sig) == 96
    assert ecrecover(sighash,
                     extract32(sig, 0, type=num256),
                     extract32(sig, 32, type=num256),
                     extract32(sig, 64, type=num256)) == self.validators[validator_index].addr
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    # Check that we are at least (epoch length / 2) blocks into the epoch
    # assert block.number % self.epoch_length >= self.epoch_length / 2
    # Check that the commit is justified
    assert self.consensus_messages[epoch].hash_justified[hash]
    # Check that this validator was active in either the previous dynasty or the current one
    ds = self.validators[validator_index].dynasty_start
    de = self.validators[validator_index].dynasty_end
    dc = self.dynasty
    in_current_dynasty = (ds <= dc) and (dc < de)
    in_prev_dynasty = (ds <= (dc - 1)) and ((dc - 1) < de)
    assert in_current_dynasty or in_prev_dynasty
    # Check that we have not yet committed for this epoch
    #assert self.validators[validator_index].max_committed == epoch - 1
    # Pay the reward if the blockhash is correct
    if True:  #if blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[validator_index].deposit * self.reward_factor)
        self.validators[validator_index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't commit for this epoch again
    self.validators[validator_index].max_committed = epoch
    # Record that this commit took place
    if in_current_dynasty:
        self.consensus_messages[epoch].commits[hash] += self.validators[validator_index].deposit
    if in_prev_dynasty:
        self.consensus_messages[epoch].prev_dyn_commits[hash] += self.validators[validator_index].deposit
    # Record if sufficient commits have been made for the block to be finalized
    if (self.consensus_messages[epoch].commits[hash] >= self.total_deposits[self.dynasty] * 2 / 3 and \
            self.consensus_messages[epoch].prev_dyn_commits[hash] >= self.total_deposits[self.dynasty - 1] * 2 / 3) and \
            not self.consensus_messages[epoch].committed:
        self.consensus_messages[epoch].committed = True

# Cannot make two prepares in the same epoch
def double_prepare_slash(validator_index: num, prepare1: bytes <= 1000, prepare2: bytes <= 1000):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash1 = extract32(raw_call(self.sighasher, prepare1, gas=200000, outsize=32), 0)
    sighash2 = extract32(raw_call(self.sighasher, prepare2, gas=200000, outsize=32), 0)
    # Extract parameters
    values1 = RLPList(prepare1, [num, bytes32, bytes32, num, bytes32, bytes])
    values2 = RLPList(prepare2, [num, bytes32, bytes32, num, bytes32, bytes])
    epoch1 = values1[0]
    sig1 = values1[5]
    epoch2 = values2[0]
    sig2 = values2[5]
    # Check the signatures
    assert ecrecover(sighash1,
                     as_num256(extract32(sig1, 0)),
                     as_num256(extract32(sig1, 32)),
                     as_num256(extract32(sig1, 64))) == self.validators[validator_index].addr
    assert ecrecover(sighash2,
                     as_num256(extract32(sig2, 0)),
                     as_num256(extract32(sig2, 32)),
                     as_num256(extract32(sig2, 64))) == self.validators[validator_index].addr
    # Check that they're from the same epoch
    assert epoch1 == epoch2
    # Check that they're not the same message
    assert sighash1 != sighash2
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= (validator_deposit - validator_deposit / 25)
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_epoch: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

def prepare_commit_inconsistency_slash(validator_index: num, prepare_msg: bytes <= 1024, commit_msg: bytes <= 1024):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash1 = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    sighash2 = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values1 = RLPList(prepare_msg, [num, bytes32, bytes32, num, bytes32, bytes])
    values2 = RLPList(commit_msg, [num, bytes32, bytes])
    prepare_epoch = values1[0]
    prepare_source_epoch = values1[3]
    sig1 = values1[5]
    commit_epoch = values2[0]
    sig2 = values2[2]
    # Check the signatures
    assert ecrecover(sighash1,
                     as_num256(extract32(sig1, 0)),
                     as_num256(extract32(sig1, 32)),
                     as_num256(extract32(sig1, 64))) == self.validators[validator_index].addr
    assert ecrecover(sighash2,
                     as_num256(extract32(sig2, 0)),
                     as_num256(extract32(sig2, 32)),
                     as_num256(extract32(sig2, 64))) == self.validators[validator_index].addr
    # Check that they're not the same message
    assert sighash1 != sighash2
    # Check that the prepare refers to something older than the commit
    assert prepare_source_epoch < commit_epoch
    # Check that the prepare is newer than the commit
    assert commit_epoch < prepare_epoch
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_epoch: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

def commit_non_justification_slash(validator_index: num, commit_msg: bytes <= 1024):
    sighash = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(commit_msg, [num, bytes32, bytes])
    epoch = values[0]
    hash = values[1]
    sig = values[2]
    # Check the signature
    assert len(sig) == 96
    assert ecrecover(sighash,
                     extract32(sig, 0, type=num256),
                     extract32(sig, 32, type=num256),
                     extract32(sig, 64, type=num256)) == self.validators[validator_index].addr
    # Check that the commit is old enough
    assert self.current_epoch == block.number / self.epoch_length
    assert (self.current_epoch - epoch) * self.epoch_length * self.block_time > self.insufficiency_slash_delay
    assert not self.consensus_messages[epoch].hash_justified[hash]
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_epoch: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }

# Fill in the table for which hash is what-degree ancestor of which other hash
def derive_parenthood(older: bytes32, hash: bytes32, newer: bytes32):
    assert sha3(concat(hash, older)) == newer
    self.ancestry[older][newer] = 1

# Fill in the table for which hash is what-degree ancestor of which other hash
def derive_ancestry(oldest: bytes32, middle: bytes32, recent: bytes32):
    assert self.ancestry[middle][recent]
    assert self.ancestry[oldest][middle]
    self.ancestry[oldest][recent] = self.ancestry[oldest][middle] + self.ancestry[middle][recent]

def prepare_non_justification_slash(validator_index: num, prepare_msg: bytes <= 1024) -> num:
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(prepare_msg, [num, bytes32, bytes32, num, bytes32, bytes])
    epoch = values[0]
    hash = values[1]
    ancestry_hash = values[2]
    source_epoch = values[3]
    source_ancestry_hash = values[4]
    sig = values[5]
    # Check the signature
    assert ecrecover(sighash,
                     extract32(sig, 0, type=num256),
                     extract32(sig, 32, type=num256),
                     extract32(sig, 64, type=num256)) == self.validators[validator_index].addr
    # Check that the view change is old enough
    assert self.current_epoch == block.number / self.epoch_length
    assert (self.current_epoch - epoch) * self.block_time * self.epoch_length > self.insufficiency_slash_delay
    # Check that the source ancestry hash not had enough prepares, OR that there is not the
    # correct ancestry link between the current ancestry hash and source ancestry hash
    c1 = self.consensus_messages[source_epoch].ancestry_hash_justified[source_ancestry_hash]
    if epoch - 1 > source_epoch:
        c2 = self.ancestry[source_ancestry_hash][ancestry_hash] == epoch - 1 - source_epoch
    else:
        c2 = source_ancestry_hash == ancestry_hash
    assert not (c1 and c2)
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        withdrawal_epoch: 0,
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
