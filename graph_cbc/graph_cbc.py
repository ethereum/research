import random

VALIDATORS = 5000
EDGES = 255
FINALITY = 4000

assert EDGES % 2 == 1

neighbors = list(range(VALIDATORS))
edgelist = neighbors * EDGES
random.shuffle(edgelist)
edges = [edgelist[i*EDGES:i*EDGES+EDGES] for i in range(VALIDATORS)]

last_votes = '1' * FINALITY + '0' * (VALIDATORS - FINALITY)

while 1:
    new_zeroes = []
    for i in range(VALIDATORS):
        votes_for_0 = len([e for e in edges[i] if last_votes[e] == '0'])
        if votes_for_0 * 2 > EDGES:
            new_zeroes.append(i)
    new_last_votes = ''.join(['01'[last_votes[j]=='1' and j not in new_zeroes]
                          for j in range(VALIDATORS)])
    print(new_last_votes.count('0'))
    if new_last_votes == last_votes:
        break
    last_votes = new_last_votes

print(last_votes.count('0'))

print("Initiating repeat-pivotal strategy")

threshold = EDGES // 2
corrupted = 0

while last_votes != '0' * VALIDATORS:
    # Attempt strategy of finding the single pivotal validator that can
    # corrupt the most other validators
    pivotals = {}
    for i in range(VALIDATORS):
        votes_for_0 = len([e for e in edges[i] if last_votes[e] == '0'])
        assert last_votes[i] == '0' or votes_for_0 * 2 < EDGES
        if votes_for_0 == threshold:
            for e in edges[i]:
                if last_votes[e] == '1':
                    pivotals[e] = pivotals.get(e, 0) + 1
    if len(pivotals) > 0:
        corrupt = [max(zip(pivotals.values(), pivotals.keys()))[1]]
    # Attempt strategy of finding the smallest group of validators that
    # can be corrupted to turn 1 more
    else:
        corrupt = []
        max_votes = 0
        for i in range(VALIDATORS):
            if last_votes[i] == '0':
                continue
            votes_for_0 = len([e for e in edges[i] if last_votes[e] == '0'])
            if votes_for_0 > max_votes:
                amount_to_corrupt = EDGES // 2 - votes_for_0 + 1
                corrupt = [e for e in edges[i] if last_votes[e] == '1'][:amount_to_corrupt]
                max_votes = votes_for_0

    last_votes = ''.join(['01'[last_votes[j]=='1' and j not in corrupt]
                          for j in range(VALIDATORS)])
    corrupted += len(corrupt)
    while 1:
        new_zeroes = []
        for i in range(VALIDATORS):
            votes_for_0 = len([e for e in edges[i] if last_votes[e] == '0'])
            if votes_for_0 * 2 > EDGES:
                new_zeroes.append(i)
        new_last_votes = ''.join(['01'[last_votes[j]=='1' and j not in new_zeroes]
                              for j in range(VALIDATORS)])
        if new_last_votes == last_votes:
            break
        last_votes = new_last_votes
    print("Corrupted %r (total %d), now on fork chain: %d" %
          (corrupt, corrupted, new_last_votes.count('0')))

print("Total corrupted: %d" % corrupted)
