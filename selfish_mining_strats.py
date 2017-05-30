import random
import sys

def test_strat(strat, hashpower, gamma, reward, fees, uncle_rewards=1, uncle_coeff=0, max_uncles=2, rounds=25000):
    # Block reward for attacker
    me_reward = 0
    # Block reward for others
    them_reward = 0
    # Fees for the attacker
    me_fees = 0
    # Fees for others
    them_fees = 0
    # Blocks in current private chain
    me_blocks = 0
    # Blocks in current public chain
    them_blocks = 0
    # Time elapsed since last chain merging
    time_elapsed = 0
    # Divisor for block rewards (diff adjustment)
    divisor = 0
    # Total blocks included from attacker
    me_totblocks = 0
    # Total blocks included from others
    them_totblocks = 0
    # Uncles included from attacker
    me_totuncles = 0
    # Uncles included from others
    them_totuncles = 0
    # Simulate the system
    for i in range(rounds):
        # Attacker makes a block
        if random.random() < hashpower:
            me_blocks += 1
            last_is_me = 1
        # Honest nodes make a block
        else:
            them_blocks += 1
            last_is_me = 0
        time_elapsed += random.expovariate(1)
        # "Adopt" or "override"
        if me_blocks >= len(strat) or them_blocks >= len(strat[me_blocks]) or strat[me_blocks][them_blocks] == 1:
            # Override
            if me_blocks > them_blocks or (me_blocks == them_blocks and random.random() < gamma):
                # me_reward += me_blocks * reward - (reward if me_blocks and them_blocks else 0)
                me_reward += me_blocks * reward
                me_fees += time_elapsed * fees
                # divisor += me_blocks - (1 if me_blocks and them_blocks else 0)
                divisor += me_blocks
                me_totblocks += me_blocks
                # Add uncles
                while me_blocks < 7 and them_blocks > 0:
                    r = min(them_blocks, max_uncles) * (0.875 - 0.125 * me_blocks) * uncle_rewards
                    them_totuncles += min(them_blocks, max_uncles)
                    divisor += min(them_blocks, max_uncles) * uncle_coeff
                    them_reward = them_reward + r
                    them_blocks -= min(them_blocks, max_uncles)
                    me_blocks += 1
            # Adopt
            else:
                # them_reward += them_blocks * reward - (reward if me_blocks and them_blocks else 0)
                them_reward += them_blocks * reward
                them_fees += time_elapsed * fees
                # divisor += them_blocks - (1 if me_blocks and them_blocks else 0)
                divisor += them_blocks
                them_totblocks += them_blocks
                # Add uncles
                while them_blocks < 7 and me_blocks > 0:
                    r = min(me_blocks, max_uncles) * (0.875 - 0.125 * them_blocks) * uncle_rewards
                    me_totuncles += min(me_blocks, max_uncles)
                    divisor += min(me_blocks, max_uncles) * uncle_coeff
                    me_reward = me_reward + r
                    me_blocks -= min(me_blocks, max_uncles)
                    them_blocks += 1
            me_blocks = 0
            them_blocks = 0
            time_elapsed = 0
        # Match
        elif strat[me_blocks][them_blocks] == 2 and not last_is_me:
            if random.random() < gamma:
                # me_reward += me_blocks * reward + time_elapsed * fees - (reward if me_blocks and them_blocks else 0)
                me_reward += me_blocks * reward
                me_totblocks += me_blocks
                # divisor += me_blocks - (1 if me_blocks and them_blocks else 0)
                divisor += me_blocks
                time_elapsed = 0
                # Add uncles
                while me_blocks < 7 and them_blocks > 0:
                    r = min(them_blocks, max_uncles) * (0.875 - 0.125 * me_blocks) * uncle_rewards
                    them_totuncles += min(them_blocks, max_uncles)
                    divisor += min(them_blocks, max_uncles) * uncle_coeff
                    them_reward = them_reward + r
                    them_blocks -= min(them_blocks, max_uncles)
                    me_blocks += 1
                me_blocks = 0
                them_blocks = 0
    # print 'rat', (me_totblocks + me_totuncles) / (me_totblocks + me_totuncles + them_totblocks + them_totuncles * 1.0)
    return me_reward / divisor + me_fees / rounds, them_reward / divisor + them_fees / rounds

# A 20x20 array meaning "what to do if I made i blocks and the network
# made j blocks?". 1 = publish, 0 = do nothing.
def gen_selfish_mining_strat():
    o = [([0] * 20) for i in range(20)]
    for me in range(20):
        for them in range(20):
            # Adopt
            if them == 1 and me == 0:
                o[me][them] = 1
            if them == me + 1:
                o[me][them] = 1
            # Overtake
            if me >= 2 and me == them + 1:
                o[me][them] = 1
            # Match
            if me >= 1 and me == them:
                o[me][them] = 2
    return o


dic = {"rewards": 1, "fees": 0, "gamma": 0.5, "uncle_coeff": 0, "uncle_rewards": 0, "max_uncles": 2}
for a in sys.argv[1:]:
    param, val = a[:a.index('=')], a[a.index('=')+1:]
    dic[param] = float(val)
print dic
s = gen_selfish_mining_strat()
for i in range(1, 50):
    x, y = test_strat(s, i * 0.01, dic["gamma"], dic["rewards"], dic["fees"],
                      dic["uncle_rewards"], dic["uncle_coeff"], dic["max_uncles"], rounds=200000)
    print '%d%% hashpower, %f%% of rewards, (%f attacker, %f honest)' % \
        (i, x * 100.0 / (x + y), x * 100.0 / i, y * 100.0 / (100-i))
