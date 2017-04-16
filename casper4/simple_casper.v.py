# Information about validators
validators: public({
    # Amount of wei the validator holds
    deposit: wei_value,
    # The dynasty the validator is joining
    dynasty_start: num,
    # The dynasty the validator joined for the first time
    original_dynasty_start: num,
    # The dynasty the validator is leaving
    dynasty_end: num,
    # The timestamp at which the validator can withdraw
    withdrawal_epoch: num,
    # The address which the validator's signatures must verify to (to be later replaced with validation code)
    addr: address,
    # Addess to withdraw to
    withdrawal_addr: address,
    # Previous epoch in which this validator committed
    prev_commit_epoch: num
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

# Mapping of epoch to what dynasty it is
dynasty_in_epoch: public(num[num])

# Information for use in processing cryptoeconomic commitments
consensus_messages: public({
    # How many prepares are there for this hash (hash of message hash + view source) from the current dynasty
    prepares: wei_value[bytes32],
    # Bitmap of which validator IDs have already prepared
    prepare_bitmap: num256[num][bytes32],
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

# A bitmap, where the ith bit of dynasty_mark[arg1][arg2] shows
# whether or not validator arg1 is active during dynasty arg2*256+i
dynasty_mask: num256[num][num]

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

# Purity checker library address
purity_checker: address

# Reward for preparing or committing, as fraction of deposit size
reward_factor: public(decimal)

# Desired total ether given out assuming 1M ETH deposited
reward_at_1m_eth: decimal

# Have I already been initialized?
initialized: bool

# Log topic for prepare
prepare_log_topic: bytes32

# Log topic for commit
commit_log_topic: bytes32

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
    self.owner = 0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6
    # Add an initial validator
    self.validators[0] = {
        deposit: as_wei_value(3, ether),
        dynasty_start: 0,
        dynasty_end: 1000000000000000000000000000000,
        original_dynasty_start: 0,
        withdrawal_epoch: 1000000000000000000000000000000,
        addr: 0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6,
        withdrawal_addr: 0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6,
        prev_commit_epoch: 0,
    }
    self.nextValidatorIndex = 1
    # Initialize the epoch counter
    self.current_epoch = block.number / self.epoch_length
    # Set the sighash calculator address
    self.sighasher = 0x476c2cA9a7f3B16FeCa86512276271FAf63B6a24
    # Set the purity checker address
    self.purity_checker = 0xD7a3BD6C9eA32efF147d067f907AE6b22d436F91
    # Set an initial root of the epoch hash chain
    self.consensus_messages[0].ancestry_hash_justified[0x0000000000000000000000000000000000000000000000000000000000000000] = True
    # self.consensus_messages[0].committed = True
    # Set initial total deposit counter
    self.total_deposits[0] = as_wei_value(3, ether)
    # Set deposit scale factor
    self.consensus_messages[0].deposit_scale_factor = 1000000000000000000.0
    # Total ETH given out assuming 1m ETH deposits
    self.reward_at_1m_eth = 12.5
    # Log topics for prepare and commit
    self.prepare_log_topic = sha3("prepare()")
    self.commit_log_topic = sha3("commit()")

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
    self.dynasty_in_epoch[epoch] = self.dynasty
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
    assert extract32(raw_call(self.purity_checker, concat('\xa1\x90>\xab', as_bytes32(validation_addr)), gas=500000, outsize=32), 0) != as_bytes32(0)
    self.validators[self.nextValidatorIndex] = {
        deposit: msg.value,
        dynasty_start: self.dynasty + 2,
        original_dynasty_start: self.dynasty + 2,
        dynasty_end: 1000000000000000000000000000000,
        withdrawal_epoch: 1000000000000000000000000000000,
        addr: validation_addr,
        withdrawal_addr: withdrawal_addr,
        prev_commit_epoch: 0,
    }
    self.nextValidatorIndex += 1
    self.second_next_dynasty_wei_delta += msg.value

# Log in or log out from the validator set. A logged out validator can log
# back in later, if they do not log in for an entire withdrawal period,
# they can get their money out
def flick_status(logout_msg: bytes <= 1024):
    assert self.current_epoch == block.number / self.epoch_length
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, logout_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(logout_msg, [num, num, bool, bytes])
    validator_index = values[0]
    epoch = values[1]
    login_flag = values[2]
    sig = values[3]
    assert self.current_epoch == epoch
    # Signature check
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash, sig), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Logging in
    if login_flag:
        # Check that we are logged out
        assert self.validators[validator_index].dynasty_end < self.dynasty
        # Check that we logged out for less than 3840 dynasties (min: ~2 months)
        assert self.validators[validator_index].dynasty_end >= self.dynasty - 3840
        # Apply the per-epoch deposit penalty
        prev_login_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_start]
        prev_logout_epoch = self.dynasty_start_epoch[self.validators[validator_index].dynasty_end + 1]
        self.validators[validator_index].deposit = \
            floor(self.validators[validator_index].deposit *
                    (self.consensus_messages[prev_logout_epoch].deposit_scale_factor /
                     self.consensus_messages[prev_login_epoch].deposit_scale_factor))
        # Log back in
        # Go through the dynasty mask to clear out the ineligible dynasties
        old_ds = self.validators[validator_index].dynasty_end
        new_ds = self.dynasty + 2
        for i in range(old_ds / 256, old_ds / 256 + 16):
            if old_ds > i * 256:
                s = old_ds % 256
            else:
                s = 0
            if new_ds < i * 256 + 256:
                e = new_ds % 256
            else:
                e = 256
            self.dynasty_mask[validator_index][i] = num256_sub(shift(as_num256(1), e), shift(as_num256(1), s))
            if e < 256:
                break
        self.validators[validator_index].dynasty_start = new_ds
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

# Removes a validator from the validator pool
def delete_validator(validator_index: num):
    self.validators[validator_index] = {
        deposit: 0,
        dynasty_start: 0,
        dynasty_end: 0,
        original_dynasty_start: 0,
        withdrawal_epoch: 0,
        addr: None,
        withdrawal_addr: None,
        prev_commit_epoch: 0,
    }

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
    self.delete_validator(validator_index)

# Checks if a given validator could have prepared in a given epoch
def check_eligible_in_epoch(validator_index: num, epoch: num) -> num(const):
    # Time limit for submitting a prepare
    assert epoch > self.current_epoch - 3840
    # Original starting dynasty of the validator; fail if before
    do = self.validators[validator_index].original_dynasty_start
    # Ending dynasty of the current login period
    de = self.validators[validator_index].dynasty_end
    # Dynasty of the prepare
    dc = self.dynasty_in_epoch[epoch]
    # Dynasty before the prepare (for prev dynasty checking)
    dp = dc - 1
    # Check against mask to see if the dynasty was eligible before
    cur_in_mask = bitwise_and(self.dynasty_mask[validator_index][dc / 256], shift(as_num256(1), dc % 256))
    prev_in_mask = bitwise_and(self.dynasty_mask[validator_index][dp / 256], shift(as_num256(1), dp % 256))
    o = 0
    # Return result as bitmask, bit 1 = in_current_dynasty, bit 0 = in_prev_dynasty
    if ((do <= dc and cur_in_mask == as_num256(0)) and dc < de):
        o += 2
    if ((do <= dp and prev_in_mask == as_num256(0)) and dp < de):
        o += 1
    return o

# Process a prepare message
def prepare(prepare_msg: bytes <= 1024):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(prepare_msg, [num, num, bytes32, bytes32, num, bytes32, bytes])
    validator_index = values[0]
    epoch = values[1]
    hash = values[2]
    ancestry_hash = values[3]
    source_epoch = values[4]
    source_ancestry_hash = values[5]
    sig = values[6]
    new_ancestry_hash = sha3(concat(hash, ancestry_hash))
    # Hash for purposes of identifying this (epoch, hash, ancestry_hash, source_epoch, source_ancestry_hash) combination
    sourcing_hash = sha3(concat(as_bytes32(epoch), hash, ancestry_hash, as_bytes32(source_epoch), source_ancestry_hash))
    # Check the signature
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash, sig), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Check that we are in an epoch after we started validating
    assert self.current_epoch >= self.dynasty_start_epoch[self.validators[validator_index].dynasty_start]
    # Check that this prepare has not yet been made
    assert not bitwise_and(self.consensus_messages[epoch].prepare_bitmap[sourcing_hash][validator_index / 256],
                           shift(as_num256(1), validator_index % 256))
    # Check that we are at least (epoch length / 4) blocks into the epoch
    # assert block.number % self.epoch_length >= self.epoch_length / 4
    # Check that this validator was active in either the previous dynasty or the current one
    epochcheck = self.check_eligible_in_epoch(validator_index, epoch)
    in_current_dynasty = epochcheck >= 2
    in_prev_dynasty = (epochcheck % 2) == 1
    assert in_current_dynasty or in_prev_dynasty
    # Check that the prepare is on top of a justified prepare
    assert self.consensus_messages[source_epoch].ancestry_hash_justified[source_ancestry_hash]
    # Check that we have not yet prepared for this epoch
    # Pay the reward if the prepare was submitted in time and the blockhash is correct
    this_validators_deposit = self.validators[validator_index].deposit
    if self.current_epoch == epoch:  #if blockhash(epoch * self.epoch_length) == hash:
        reward = floor(this_validators_deposit * self.reward_factor)
        self.validators[validator_index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't prepare for this epoch again
    self.consensus_messages[epoch].prepare_bitmap[sourcing_hash][validator_index / 256] = \
        bitwise_or(self.consensus_messages[epoch].prepare_bitmap[sourcing_hash][validator_index / 256],
                   shift(as_num256(1), validator_index % 256))
    # self.validators[validator_index].max_prepared = epoch
    # Record that this prepare took place
    curdyn_prepares = self.consensus_messages[epoch].prepares[sourcing_hash]
    if in_current_dynasty:
        curdyn_prepares += this_validators_deposit
        self.consensus_messages[epoch].prepares[sourcing_hash] = curdyn_prepares
    prevdyn_prepares = self.consensus_messages[epoch].prev_dyn_prepares[sourcing_hash]
    if in_prev_dynasty:
        prevdyn_prepares += this_validators_deposit
        self.consensus_messages[epoch].prev_dyn_prepares[sourcing_hash] = prevdyn_prepares
    # If enough prepares with the same epoch_source and hash are made,
    # then the hash value is justified for commitment
    if (curdyn_prepares >= self.total_deposits[self.dynasty] * 2 / 3 and \
            prevdyn_prepares >= self.total_deposits[self.dynasty - 1] * 2 / 3) and \
            not self.consensus_messages[epoch].ancestry_hash_justified[new_ancestry_hash]:
        self.consensus_messages[epoch].ancestry_hash_justified[new_ancestry_hash] = True
        self.consensus_messages[epoch].hash_justified[hash] = True
    # Add a parent-child relation between ancestry hashes to the ancestry table
    if not self.ancestry[ancestry_hash][new_ancestry_hash]:
        self.ancestry[ancestry_hash][new_ancestry_hash] = 1
    raw_log([self.prepare_log_topic], prepare_msg)

# Process a commit message
def commit(commit_msg: bytes <= 1024):
    sighash = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(commit_msg, [num, num, bytes32, num, bytes])
    validator_index = values[0]
    epoch = values[1]
    hash = values[2]
    prev_commit_epoch = values[3]
    sig = values[4]
    # Check the signature
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash, sig), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    # Check that we are at least (epoch length / 2) blocks into the epoch
    # assert block.number % self.epoch_length >= self.epoch_length / 2
    # Check that the commit is justified
    assert self.consensus_messages[epoch].hash_justified[hash]
    # Check that this validator was active in either the previous dynasty or the current one
    epochcheck = self.check_eligible_in_epoch(validator_index, epoch)
    in_current_dynasty = epochcheck >= 2
    in_prev_dynasty = (epochcheck % 2) == 1
    assert in_current_dynasty or in_prev_dynasty
    # Check that we have not yet committed for this epoch
    assert self.validators[validator_index].prev_commit_epoch == prev_commit_epoch
    assert prev_commit_epoch < epoch
    self.validators[validator_index].prev_commit_epoch = epoch
    this_validators_deposit = self.validators[validator_index].deposit
    # Pay the reward if the blockhash is correct
    if True:  #if blockhash(epoch * self.epoch_length) == hash:
        reward = floor(this_validators_deposit * self.reward_factor)
        self.validators[validator_index].deposit += reward
        self.total_deposits[self.dynasty] += reward
    # Can't commit for this epoch again
    # self.validators[validator_index].max_committed = epoch
    # Record that this commit took place
    if in_current_dynasty:
        self.consensus_messages[epoch].commits[hash] += this_validators_deposit
    if in_prev_dynasty:
        self.consensus_messages[epoch].prev_dyn_commits[hash] += this_validators_deposit
    # Record if sufficient commits have been made for the block to be finalized
    if (self.consensus_messages[epoch].commits[hash] >= self.total_deposits[self.dynasty] * 2 / 3 and \
            self.consensus_messages[epoch].prev_dyn_commits[hash] >= self.total_deposits[self.dynasty - 1] * 2 / 3) and \
            not self.consensus_messages[epoch].committed:
        self.consensus_messages[epoch].committed = True
    raw_log([self.commit_log_topic], commit_msg)

# Cannot make two prepares in the same epoch
def double_prepare_slash(prepare1: bytes <= 1000, prepare2: bytes <= 1000):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash1 = extract32(raw_call(self.sighasher, prepare1, gas=200000, outsize=32), 0)
    sighash2 = extract32(raw_call(self.sighasher, prepare2, gas=200000, outsize=32), 0)
    # Extract parameters
    values1 = RLPList(prepare1, [num, num, bytes32, bytes32, num, bytes32, bytes])
    values2 = RLPList(prepare2, [num, num, bytes32, bytes32, num, bytes32, bytes])
    validator_index = values1[0]
    epoch1 = values1[1]
    sig1 = values1[6]
    assert validator_index == values2[0]
    epoch2 = values2[1]
    sig2 = values2[6]
    # Check the signatures
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash1, sig1), gas=500000, outsize=32), 0) == as_bytes32(1)
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash2, sig2), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Check that they're from the same epoch
    assert epoch1 == epoch2
    # Check that they're not the same message
    assert sighash1 != sighash2
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= (validator_deposit - validator_deposit / 25)
    self.delete_validator(validator_index)

def prepare_commit_inconsistency_slash(prepare_msg: bytes <= 1024, commit_msg: bytes <= 1024):
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash1 = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    sighash2 = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values1 = RLPList(prepare_msg, [num, num, bytes32, bytes32, num, bytes32, bytes])
    values2 = RLPList(commit_msg, [num, num, bytes32, num, bytes])
    validator_index = values1[0]
    prepare_epoch = values1[1]
    prepare_source_epoch = values1[4]
    sig1 = values1[6]
    assert validator_index == values2[0]
    commit_epoch = values2[1]
    sig2 = values2[4]
    # Check the signatures
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash1, sig1), gas=500000, outsize=32), 0) == as_bytes32(1)
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash2, sig2), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Check that the prepare refers to something older than the commit
    assert prepare_source_epoch < commit_epoch
    # Check that the prepare is newer than the commit
    assert commit_epoch < prepare_epoch
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.delete_validator(validator_index)

def commit_non_justification_slash(commit_msg: bytes <= 1024):
    sighash = extract32(raw_call(self.sighasher, commit_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(commit_msg, [num, num, bytes32, num, bytes])
    validator_index = values[0]
    epoch = values[1]
    hash = values[2]
    sig = values[4]
    # Check the signature
    assert len(sig) == 96
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash, sig), gas=500000, outsize=32), 0) == as_bytes32(1)
    # Check that the commit is old enough
    assert self.current_epoch == block.number / self.epoch_length
    assert (self.current_epoch - epoch) * self.epoch_length * self.block_time > self.insufficiency_slash_delay
    assert not self.consensus_messages[epoch].hash_justified[hash]
    # Delete the offending validator, and give a 4% "finder's fee"
    validator_deposit = self.validators[validator_index].deposit
    send(msg.sender, validator_deposit / 25)
    self.total_destroyed += validator_deposit * 24 / 25
    self.total_deposits[self.dynasty] -= validator_deposit
    self.delete_validator(validator_index)

# Fill in the table for which hash is what-degree ancestor of which other hash
def derive_parenthood(older: bytes32, hash: bytes32, newer: bytes32):
    assert sha3(concat(hash, older)) == newer
    self.ancestry[older][newer] = 1

# Fill in the table for which hash is what-degree ancestor of which other hash
def derive_ancestry(oldest: bytes32, middle: bytes32, recent: bytes32):
    assert self.ancestry[middle][recent]
    assert self.ancestry[oldest][middle]
    self.ancestry[oldest][recent] = self.ancestry[oldest][middle] + self.ancestry[middle][recent]

def prepare_non_justification_slash(prepare_msg: bytes <= 1024) -> num:
    # Get hash for signature, and implicitly assert that it is an RLP list
    # consisting solely of RLP elements
    sighash = extract32(raw_call(self.sighasher, prepare_msg, gas=200000, outsize=32), 0)
    # Extract parameters
    values = RLPList(prepare_msg, [num, num, bytes32, bytes32, num, bytes32, bytes])
    validator_index = values[0]
    epoch = values[1]
    hash = values[2]
    ancestry_hash = values[3]
    source_epoch = values[4]
    source_ancestry_hash = values[5]
    sig = values[6]
    # Check the signature
    assert extract32(raw_call(self.validators[validator_index].addr, concat(sighash, sig), gas=500000, outsize=32), 0) == as_bytes32(1)
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
    self.delete_validator(validator_index)

# Temporary backdoor for testing purposes (to allow recovering destroyed deposits)
def owner_withdraw():
    send(self.owner, self.total_destroyed)
    self.total_destroyed = 0

# Change backdoor address (set to zero to remove entirely)
def change_owner(new_owner: address):
    if self.owner == msg.sender:
        self.owner = new_owner
