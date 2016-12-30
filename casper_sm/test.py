# The purpose of this script is to test selfish-mining-like strategies
# in the randao-based single-chain Casper.
import random

# Penalty for mining a dunkle
DUNKLE_PENALTY = 1
# Penalty to a main-chain block which has a dunkle as a sister
DUNKLE_SISTER_PENALTY = 0.8

# Attacker stake power (out of 100). Try setting this value to any
# amount, even values above 50!
attacker_share = 40

# A simulated Casper randao
def randao_successor(parent, index):
    return (((parent ^ 53) + index) ** 3) % (10**20 - 11)

# We categorize "scenarios" by seeing how far ahead we get a chain
# of 0-skips from each "path"; if the path itself isn't clear then
# return zero
heads_of_interest = ['', '1']
# Only scan down this far
scandepth = 4
# eg. (0, 2) means "going straight from the current randao, you
# can descend zero, but if you make a one-skip block, from there
# you get two 0-skips in a row"
scenarios = [None, (0, 1), (0, 2), (0, 3), (0, 4)]
# For each scenario, this is the corresponding "path" to go down
paths = ['', '10', '100', '100', '1000']

# Determine the scenario ID (zero is catch-all) from a chain
def extract_scenario(chain):
    chain = chain.copy()
    o = []
    for h in heads_of_interest:
        # Make sure that we can descend down "the path"
        succeed = True
        for step in h:
            if not chain.can_i_extend(int(step)):
                succeed = False
                break
            chain.extend_me(int(step))
        if not succeed:
            o.append(0)
        else:
            # See how far down we can go
            i = 0
            while chain.can_i_extend(0) and i < scandepth:
                i += 1
                chain.extend_me(0)
            o.append(i)
    if tuple(o) in scenarios:
        return scenarios.index(tuple(o))
    else:
        return 0

# Class to represent simulated chains
class Chain():
    def __init__(self, randao=0, time=0, length=0, me=0, them=0):
        self.randao = randao
        self.time = time
        self.length = length
        self.me = me
        self.them = them

    def copy(self):
        return Chain(self.randao, self.time, self.length, self.me, self.them)

    def can_i_extend(self, skips):
        return randao_successor(self.randao, skips) % 100 < attacker_share

    def can_they_extend(self, skips):
        return randao_successor(self.randao, skips) % 100 >= attacker_share

    def extend_me(self, skips):
        new_randao = randao_successor(self.randao, skips)
        assert new_randao % 100 < attacker_share
        self.randao = new_randao
        self.time += skips
        self.length += 1
        self.me += 1

    def extend_them(self, skips):
        new_randao = randao_successor(self.randao, skips)
        assert new_randao % 100 >= attacker_share
        self.randao = new_randao
        self.time += skips
        self.length += 1
        self.them += 1

    def add_my_dunkles(self, n):
        self.me -= n * DUNKLE_PENALTY
        self.them -= n * DUNKLE_SISTER_PENALTY

    def add_their_dunkles(self, n):
        self.them -= n * DUNKLE_PENALTY
        self.me -= n * DUNKLE_SISTER_PENALTY



for strat_id in range(2**len(scenarios)):
    # Strategy map: scenario to 0 = publish, 1 = selfish-validate
    strategy = [0] + [((strat_id // 2**i) % 2) for i in range(1, len(scenarios))]
    # 1 = once we go through the selfish-validating "path", reveal it instantly
    # 0 = don't reveal until the "main chain" looks like it's close to catching up
    insta_reveal = strat_id % 2

    print 'Testing strategy: %r, insta_reveal: %d', (strategy, insta_reveal)

    pubchain = Chain(randao=random.randrange(10**20))

    time = 0
    while time < 100000:
        # You honestly get a block
        if pubchain.can_i_extend(0):
            pubchain.extend_me(0)
            time += 1
            continue
        e = extract_scenario(pubchain)
        if strategy[e] == 0:
            # You honestly let them get a block
            pubchain.extend_them(0)
            time += 1
            continue
        # Build up the secret chain based on the detected path
        # print 'Selfish mining along path %r' % paths[e]
        old_me = pubchain.me
        old_them = pubchain.them
        old_time = time
        secchain = pubchain.copy()
        sectime = time
        for skipz in paths[e]:
            skips = int(skipz)
            secchain.extend_me(skips)
            sectime += skips + 1
        # Public chain builds itself up in the meantime
        pubwait = 0
        while time < sectime:
            if pubchain.can_they_extend(pubwait):
                pubchain.extend_them(pubwait)
                pubwait = 0
            else:
                pubwait += 1
            time += 1
        secwait = 0
        # If the two chains have equal length, or if the secret chain is more than 1 longer, they duel
        while (secchain.length > pubchain.length + 1 or secchain.length == pubchain.length) and time < 100000 and not insta_reveal:
            if pubchain.can_they_extend(pubwait):
                pubchain.extend_them(pubwait)
                pubwait = 0
            else:
                pubwait += 1
            if secchain.can_i_extend(secwait):
                secchain.extend_me(secwait)
                secwait = 0
            else:
                secwait += 1
            time += 1
        # Secret chain is longer, takes over public chain, public chain goes in as dunkles
        if secchain.length > pubchain.length:
            pubchain_blocks = pubchain.them - old_them
            assert pubchain.me == old_me
            pubchain = secchain
            pubchain.add_their_dunkles(pubchain_blocks)
        # Public chain is longer, miner deletes secret chain so no dunkling
        else:
            pass
        # print 'Score deltas: me %.2f them %.2f, time delta %d' % (pubchain.me - old_me, pubchain.them - old_them, time - old_time) 

    gf = (pubchain.them - 100000. * (100 - attacker_share) / 100) / (pubchain.me - 100000 * attacker_share / 100)
    print 'My revenue: %d, their revenue: %d, griefing factor %.2f' % (pubchain.me, pubchain.them, gf)
