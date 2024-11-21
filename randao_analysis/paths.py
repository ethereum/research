import random
from heapq import heappush, heappop, heapify

LATENCY = 1
SKIP = 6

# The score of a node in the search graph, for A* search
def score_node(n):
    time, height = n
    return time - height * (LATENCY * 5.0 + 0.0000001)

# `alpha`: percentage of the network controlled by the actor
# `max_height`: stop at this height
# Searches for the shortest path (in terms of time) to
# reach any particular height, and builds up an array
# `outputs`, mapping height => min time needed to reach that
# height
def search(alpha, max_height):
    # Nodes are of the form (time, height)
    first_node = (0, 0)
    best_score = score_node(first_node)
    # The heap of nodes in the search graph (ie. blocks)
    nodes = [(best_score, first_node)]
    # Minimum time needed to reach a given height
    outputs = [0]
    # For efficiency reasons, create a random list of delays,
    # instead of recalculating these every time
    delays = []
    for i in range(200):
        o = [LATENCY]
        for _ in range(3):
            while random.random() > alpha:
                o[-1] += SKIP
            o.append(o[-1])
        delays.append(o)

    while len(nodes):
        # Process the most favorable not-yet-processed node
        score, (time, height) = heappop(nodes)
        # If this is the first time we've processed a node
        # with this height...
        if height > len(outputs):
            outputs.append(time)
            # Check if we should exit
            if height >= max_height:
                return outputs
        # Add a maximum of 3 children to this node
        for v in random.choice(delays):
            newnode = (time + v, height + 1)
            heappush(nodes, (score_node(newnode), newnode))
        # For efficiency: cap the heap size at 500-1000
        if len(nodes) > 1000:
            nodes = nodes[:500]

# Generate an array, height => time when an honest
# chain reaches that height
def honest_chain(alpha, maxheight):
    outputs = [0]
    for i in range(maxheight):
        t = outputs[-1]
        t += LATENCY
        while random.random() > alpha:
            t += SKIP
        outputs.append(t)
    return outputs

# Gets the height => time array for honest and
# attacking party, and returns the highest lead
# that the attacker gets over the honest party
def get_attacker_lead(honest, attacker):
    maxlead = -1
    for i in range(len(honest)):
        for j in range(i + maxlead + 1, len(attacker)):
            if attacker[j] < honest[i]:
                maxlead = j-i
            else:
                break
    return maxlead + 1

# Attempts a race between the attacker and honest chain.
# `attacker` = attacker share of network
# `honest` = honest share of network
# `maxheight` = give up here
# Returns the max lead that the attacker gets;
# alternatively this could be interpreted as the max
# handicap the attacker could overcome
def race(attacker, honest, maxheight):
    h = honest_chain(honest, maxheight)
    a = search(attacker, maxheight)
    return get_attacker_lead(h, a)

# Probability that an attacker will win in a standard PoW-style model
def standard_race_prob(attacker, honest, handicap):
    # Explanation for this formula is that it's the unique (sensible)
    # solution to the infinite system of equations that defines it:
    #
    # P(a, h, k) = a/(a+h) * P(a, h, k-1) + h/(a+h) * P(a, h, k+1)
    #
    # For all k (there's also the boundary condition P(a, h, 0) = 1)
    #
    # This equation arises because from any given state, there's a
    # a/(a+h) chance the attacker makes the next block, and a 
    # h/(a+h) chance the honest nodes make the next block. To see how
    # the formula below solves this, let the result for handicap=k be
    # N; note the result for k+1 is N*a/h, and for k-1 is N*h/a. Now:
    # 
    # N = a/(a+h) * N*h/a + h/(a+h) * N*a/h
    # 1 = a/(a+h) * h/a + h/(a+h) * a/h
    # 1 = h/(a+h) + a/(a+h)

    return (attacker / honest) ** handicap

# The percentage that would be required to get a given success rate
# with a given handicap
def standard_race_equiv_rate(probability, handicap):
    y = probability ** (1/handicap)
    return y/(1+y)

def test():
    # Assume attacker has (0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28); check race probs for handicap 1-8
    print("Basic tests\n")
    for alpha in (0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28):
        race_results = [race(alpha, 1-alpha, 1000) for j in range(100)]
        probs = []
        for i in range(1, 9):
            probs.append(len([x for x in race_results if x >= i]) / 100)
        print("Probs at %.2f: %r" % (alpha, probs))
        print("Standard race equiv rate: %r" % [standard_race_equiv_rate(x, i+1) for i, x in enumerate(probs)])
    # Now, try the version where each block requires notarizations from an extra two validators. Here,
    # if either side has share p, they effectively "really" have share p**3
    print("\nRequiring 2/2 notarization\n")
    for alpha in (0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39):
        race_results = [race(alpha ** 3, (1-alpha) ** 3, 1000) for j in range(100)]
        probs = []
        for i in range(1, 9):
            probs.append(len([x for x in race_results if x >= i]) / 100)
        print("Probs at %.2f: %r" % (alpha, probs))
        print("Standard race equiv rate: %r" % [standard_race_equiv_rate(x, i+1) for i, x in enumerate(probs)])

if __name__ == '__main__':
    test()
