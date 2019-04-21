import copy, os, random, binascii

zero_hash = '00000000'

def new_hash():
    return binascii.hexlify(os.urandom(4)).decode('utf-8')

# Account state record
class Account():
    def __init__(self, yes_dep, no_deps, balance):
        # This state is conditional on this dependency being CORRECT
        self.yes_dep = yes_dep
        # This state is conditional on these dependencies being INCORRECT
        self.no_deps = no_deps
        # Account balance
        self.balance = balance

    def __repr__(self):
        return "[yes_dep: %r, no_deps: %r, balance: %d]" % (self.yes_dep, self.no_deps, self.balance)

# Dependency object: (hash, height)
class Dependency():
    def __init__(self, height, hash):
        self.height = height
        self.hash = hash

    def __repr__(self):
        return "[height: %d, hash: %s]" % (self.height, self.hash)

# Test with 5 accounts
actors = ["Alice", "Bob", "Charlie", "David", "Epsie"]

# Initial empty state
state = {a: Account(Dependency(0, zero_hash), [], 100) for a in actors}

# The set of dependencies that the protocol thinks is most likely to be correct
# Think of these as being receipt roots of all shards, with the i'th hash
# corresponding to the receipt root from the i'th slot. This list gets extended
# as the protocol finds out about new dependencies, and dependencies can get popped
# off the end if the protocol realizes that they are incorrect
deps = [zero_hash]

# Alternate states corresponding to dependencies other than those the protocol thinks
# is most likely to be correct. In a real implementation these can be stored as
# receipts rather than state objects if desired
alt_states = {a: [] for a in actors}

# Does the current account state match the dependencies?
def is_account_state_active(account):
    for dep in account.no_deps:
        if (dep.height < len(deps) and dep.hash == deps[dep.height]):
            return False
    yes_dep_ht = account.yes_dep.height
    if yes_dep_ht >= len(deps) or account.yes_dep.hash != deps[yes_dep_ht]:
        return False
    return True

# Set the main account state to be the correct one
def reorg(address):
    if is_account_state_active(state[address]):
        return
    for i,r in enumerate(alt_states[address]):
        if is_account_state_active(r):
            state[address], alt_states[address][i] = alt_states[address][i], state[address]
            return
    raise Exception("wtf m8")

# Adjusts balance of an account. UNSAFE unless checks are done!
def balance_delta(to, value, height, hash):
    if height > state[to].yes_dep.height:
        alt_states[to].append(Account(
            state[to].yes_dep,
            state[to].no_deps + [Dependency(height, hash)],
            state[to].balance
        ))
    state[to].yes_dep = Dependency(height, hash)
    state[to].balance += value
    assert state[to].balance >= 0
   
# Transfer from an account (or from the outside) to another account
def transfer(frm, to, value, height, hash):
    # print("<------- Xferring", frm, to, value, height, hash)
    if len(deps) <= height or deps[height] != hash:
        return False
    if not (is_account_state_active(state[to]) and height >= state[to].yes_dep.height):
        return False
    if frm is not None and not (is_account_state_active(state[frm]) and height >= state[frm].yes_dep.height):
        return False
    balance_delta(to, value, height, hash)
    if frm is not None:
        balance_delta(frm, -value, height, hash)
    return True

print("Starting balance: %d" % state["Charlie"].balance)

# Run the main test....
for i in range(200):
    r = random.random()
    # 16% chance: new dependency
    if r < 0.16:
        deps.append(new_hash())
    # 20% chance: xfer from another shard
    elif r < 0.36:
        to = random.choice(actors)
        reorg(to)
        value = random.randrange(100)
        assert transfer(None, to, value, len(deps)-1, deps[-1])
        if to == "Charlie":
            print("Received %d coins, new balance %d, conditional on (%d, %s)" % (value, state[to].balance, len(deps)-1, deps[-1]))
    # 60% chance: xfer between two accounts inside the shard
    elif r < 0.96:
        to = random.choice(actors)
        frm = random.choice([x for x in actors if x != to])
        reorg(to)
        reorg(frm)
        common_ht = max(state[to].yes_dep.height, state[frm].yes_dep.height)
        value = random.randrange(state[frm].balance + 1)
        assert transfer(frm, to, value, common_ht, deps[common_ht])
        if frm == "Charlie":
            print("Sent %d coins, new balance %d, conditional on (%d, %s)" % (value, state[frm].balance, common_ht, deps[common_ht]))
        if to == "Charlie":
            print("Received %d coins from %s, new balance %d, conditional on (%d, %s)" % (value, frm, state[to].balance, common_ht, deps[common_ht]))
    # 4% chance: revert some dependencies
    else:
        num_to_revert = min(random.randrange(8), len(deps) - 1)
        for i in range(num_to_revert):
            print("Reverted (%d, %s)" % (len(deps)-1, deps.pop()))
        for i in range(num_to_revert):
            deps.append(new_hash())
        reorg("Charlie")
        print("New balance: %d" % state["Charlie"].balance)
