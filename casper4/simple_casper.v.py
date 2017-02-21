# Information about validators
validators: {
    # Amount of wei the validator holds
    deposit: wei_value,
    # The epoch the validator is joining
    epoch_start: num,
    # The epoch the validator is leaving
    epoch_end: num,
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

# Information for use in processing cryptoeconomic commitments
consensus_messages: {
    # Total deposits during this epoch
    total_deposits: wei_value,
    # How many prepares are there for this hash (hash of message hash + view source)
    prepares: wei_value[bytes32],
    # Is a commit on the given hash justified?
    justified: bool[bytes32],
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

def initialize_epoch(epoch: num):
    computed_current_epoch = block.number / self.epoch_length
    if epoch <= computed_current_epoch and epoch == self.current_epoch + 1:
        self.consensus_messages[epoch].total_deposits += self.consensus_messages[self.current_epoch].total_deposits
        self.current_epoch = epoch

def deposit(validation_addr: address, withdrawal_addr: address):
    assert self.current_epoch == block.number / self.epoch_length
    start_epoch = (self.current_epoch - (self.current_epoch % 50)) + 100
    self.validators[self.nextValidatorIndex] = {
        deposit: msg.value,
        epoch_start: start_epoch,
        epoch_end: 1000000000000000000000000000000,
        withdrawal_time: 1000000000000000000000000000000,
        addr: validation_addr,
        withdrawal_addr: withdrawal_addr,
        max_prepared: 0,
        max_committed: 0,
    }
    self.nextValidatorIndex += 1
    self.interest_rate = 0.000001
    self.block_time = 7
    self.epoch_length = 1024
    self.withdrawal_delay = 2500000
    self.insufficiency_slash_delay = self.withdrawal_delay / 2
    self.consensus_messages[start_epoch].total_deposits += msg.value

def start_withdrawal(index: num, sig: bytes <= 96):
    # Signature check
    assert len(sig) == 96
    assert ecrecover(sha3("withdraw"),
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that we haven't already withdrawn
    assert self.current_epoch == block.number / self.epoch_length
    end_epoch = (self.current_epoch - (self.current_epoch % 50)) + 100
    assert self.validators[index].epoch_end > end_epoch
    # Set the end epoch
    self.validators[index].epoch_end = end_epoch
    self.consensus_messages[end_epoch].total_deposits -= self.validators[index].deposit
    # Set the withdrawal date
    self.validators[index].withdrawal_time = block.timestamp + self.withdrawal_delay

def withdraw(index: num):
    if block.timestamp >= self.validators[index].withdrawal_time:
        send(self.validators[index].withdrawal_addr, self.validators[index].deposit)
        self.validators[index] = {
            deposit: 0,
            epoch_start: 0,
            epoch_end: 0,
            withdrawal_time: 0,
            addr: None,
            withdrawal_addr: None,
            max_prepared: 0,
            max_committed: 0,
        }

def prepare(index: num, epoch: num, hash: bytes32, ancestry_hash: bytes32,
            epoch_source: num, source_ancestry_hash: bytes32, sig: bytes <= 96):
    # Signature check
    sighash = sha3(concat("prepare", as_bytes32(epoch), hash, ancestry_hash, as_bytes32(epoch_source), source_ancestry_hash))
    assert len(sig) == 96
    assert ecrecover(sighash,
                     as_num256(extract32(sig, 0)),
                     as_num256(extract32(sig, 32)),
                     as_num256(extract32(sig, 64))) == self.validators[index].addr
    # Check that we are in the right epoch
    assert self.current_epoch == block.number / self.epoch_length
    assert self.current_epoch == epoch
    assert self.validators[index].epoch_start <= epoch
    assert epoch < self.validators[index].epoch_end
    # Check that we have not yet prepared for this epoch
    assert self.validators[index].max_prepared == epoch - 1
    # Pay the reward if the blockhash is correct
    if True: #~blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[index].deposit * self.interest_rate * self.block_time)
        self.validators[index].deposit += reward
        self.consensus_messages[self.current_epoch].total_deposits += reward
    # Can't prepare for this epoch again
    self.validators[index].max_prepared = epoch
    # Record that this prepare took place
    new_ancestry_hash = sha3(concat(hash, ancestry_hash))
    self.consensus_messages[epoch].prepares[sighash] += self.validators[index].deposit
    # If enough prepares with the same epoch_source and hash are made,
    # then the hash value is justified for commitment
    if self.consensus_messages[epoch].prepares[sighash] >= self.consensus_messages[epoch].total_deposits * 2 / 3 and \
            not self.consensus_messages[epoch].justified[new_ancestry_hash]:
        self.consensus_messages[epoch].justified[new_ancestry_hash] = True
    # Add a parent-child relation between ancestry hashes to the ancestry table
    self.ancestry[ancestry_hash][new_ancestry_hash] = 1

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
    assert self.validators[index].epoch_start <= epoch
    assert epoch < self.validators[index].epoch_end
    # Check that we have not yet committed for this epoch
    assert self.validators[index].max_committed == epoch - 1
    # Pay the reward if the blockhash is correct
    if True: #~blockhash(epoch * self.epoch_length) == hash:
        reward = floor(self.validators[index].deposit * self.interest_rate * self.block_time)
        self.validators[index].deposit += reward
        self.consensus_messages[self.current_epoch].total_deposits += reward
    # Can't commit for this epoch again
    self.validators[index].max_committed = epoch

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
    self.consensus_messages[self.current_epoch].total_deposits -= (validator_deposit - validator_deposit / 25)
    self.validators[index] = {
        deposit: 0,
        epoch_start: 0,
        epoch_end: 0,
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
    self.consensus_messages[self.current_epoch].total_deposits -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        epoch_start: 0,
        epoch_end: 0,
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
    self.consensus_messages[self.current_epoch].total_deposits -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        epoch_start: 0,
        epoch_end: 0,
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
    self.consensus_messages[self.current_epoch].total_deposits -= validator_deposit
    self.validators[index] = {
        deposit: 0,
        epoch_start: 0,
        epoch_end: 0,
        withdrawal_time: 0,
        addr: None,
        withdrawal_addr: None,
        max_prepared: 0,
        max_committed: 0,
    }
