import random
# Attacker percent of hashpower
alpha = 0.333
# Probability attacker can win in a network race
gamma = 0.0

# Total blocks of attacker and honest miners
attacker_blocks = 0
honest_blocks = 0
# At the start of each outer loop, the attacker and honest
# chains are fully shared
while attacker_blocks + honest_blocks < 100000:
    # 1-alpha: honest miner makes block, attacker accepts
    if random.random() > alpha:
        honest_blocks += 1
        continue
    # alpha: attacker makes block

    # 1-alpha: honest miners publish a block and catch up
    # Attacker publishes their block. They now have two
    # chances to win: (i) they make the next block, (ii)
    # the next block is made by an honest miner who saw
    # the attacker's block first
    if random.random() > alpha:
        # Alpha: attacker mines next block, publishes both blocks
        if random.random() < alpha:
            attacker_blocks += 2
            continue
        # 1-alpha: honest miners mine next block, publish
        # gamma: that honest miner saw the attacker's block first,
        # so the new winning chain contains one honest block and
        # one attacker block
        if random.random() < gamma:
            honest_blocks += 1
            attacker_blocks += 1
        # 1-gamma: that honest miner saw the honest block first,
        # so the winning chain contains two honest blocks
        else:
            honest_blocks += 2
        continue
    # alpha: attacker is now two blocks ahead
    state = [2, 0]

    # Keep going until attacker only one block ahead
    while state[0] - state[1] > 1:
        if random.random() > alpha:
            state[1] += 1
        else:
            state[0] += 1
    # Attacker publishes, overtakes honest chain
    attacker_blocks += state[0]

print("Attacker blocks percentage: %.3f" %
      (attacker_blocks / (attacker_blocks + honest_blocks) * 100))
