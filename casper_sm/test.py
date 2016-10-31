import random

CHECK_DEPTH = 4
BLOCK_TIME = 1
SKIP_TIME = 1
FORK_PUNISHMENT_COEFF = 1.5

attacker_share = 0.45


# randao_results[1101] = 1 -> if validator path is 1, 0, 1 then
# a colluding node will make the next block. 
# randao_results[10011] = 0 -> if validator path is 1, 1, 0, 0
# then a non-colluding node will make the next block.
randao_results = [None, None]
for i in range(2, 2**CHECK_DEPTH):
    randao_results.append(1 if random.random() < attacker_share else 0)

def update_randao():
    for i in range(2, len(randao_results) / 2):
        randao_results[i] = randao_results[i * 2]
    for i in range(len(randao_results) / 2, len(randao_results)):
        randao_results[i] = 1 if random.random() < attacker_share else 0

# Strategy map: number of blocks belonging to colluding node in
# the 0, 0, 0... chain -> 0 = publish, 1 = wait and compete, 2 = don't
# publish

for strat_id in range(2**CHECK_DEPTH):
    strategy = [0]
    k = strat_id
    for _ in range(CHECK_DEPTH):
        strategy.append(k % 2)
        k /= 2

    my_revenue = 0.
    their_revenue = 0.

    print 'Testing strategy:', strategy

    time = 0
    while time < 200000:
        child_0 = randao_results[2]
        child_1 = randao_results[3]
        situation = 0
        q = 2
        while situation < len(strategy) - 2 and randao_results[q] == 1:
            situation += 1
            q *= 2
        if child_0 == 0:
            their_revenue += 1
            time += BLOCK_TIME
        elif strategy[situation] == 0:
            my_revenue += 1
            time += BLOCK_TIME
        elif strategy[situation] == 2:
            their_revenue += 1
            time += BLOCK_TIME + SKIP_TIME
        else:
            assert situation >= 1
            my_score = situation
            my_time = situation + 1
            their_time = 0
            their_score = 0
            update_randao()
            while their_time <= situation:
                if randao_results[2] == 0:
                    wait_time = BLOCK_TIME
                elif randao_results[3] == 0:
                    wait_time = BLOCK_TIME + SKIP_TIME
                else:
                    wait_time = BLOCK_TIME + SKIP_TIME * 2
                    while random.random() < attacker_share:
                        wait_time += SKIP_TIME
                their_score += 1
                their_time += wait_time
                update_randao()
            time += max(my_time, their_time)
            while my_score > their_score + 1:
                if random.random() < attacker_share:
                    my_score += 1
                if random.random() > attacker_share:
                    their_score += 1
                time += BLOCK_TIME
            if my_score > their_score or (my_score == their_score and random.random() < 0.5):
                my_revenue += my_score - FORK_PUNISHMENT_COEFF
                their_revenue -= their_score * 0.5
            else:
                their_revenue += their_score - FORK_PUNISHMENT_COEFF
                my_revenue -= my_score * 0.5
        update_randao()
        
    if my_revenue * 1.0 / (my_revenue + their_revenue) > attacker_share - 0.01:
        print 'My revenue:', my_revenue / time
        print 'Their revenue:', their_revenue / time
        print 'My share:', my_revenue / (my_revenue + their_revenue)
        print 'Griefing factor:', (their_revenue / time / (1 - attacker_share) - 1) / (my_revenue / time / attacker_share - 1)
