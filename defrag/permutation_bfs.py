import heapq, random

def distance_score(state):
    state = [int(x) for x in state.split(',')]
    tot = 0
    for i, s in enumerate(state):
        xorval = (i+1) ^ s
        indexmask = 1
        while indexmask < len(state):
            if xorval & indexmask:
                tot += 1
            indexmask <<= 1
    return tot

def generate_legal_moves_for_bit(state, bit):
    o = set()
    xormask = 2**bit
    state = [int(x) for x in state.split(',')]
    for i in range(2**(len(state)-1)):
        new_state = state[::]
        indexmask = 1
        for j in range(len(state)):
            if j > (j ^ xormask):
                if i & indexmask:
                    new_state[j], new_state[j ^ xormask] = new_state[j ^ xormask], new_state[j]
                indexmask <<= 1
        o.add(','.join([str(x) for x in new_state]))
    return o

def generate_legal_moves(state):
    o = set()
    b = 0
    while 2**b < len(state):
        o = o.union(generate_legal_moves_for_bit(state, b))
        b += 1
    return o

def mk_shuffle(n):
    L = list(range(1, n+1))
    random.shuffle(L)
    return ','.join([str(x) for x in L])

def find_path(start):
    parents = {}
    scores = {start: 0}
    queue = [(distance_score(start), start)]
    goal = ','.join([str(x) for x in sorted([int(x) for x in start.split(',')])])
    totvs = 0
    while len(queue):
        qval = heapq.heappop(queue)[1]
        newvals = [x for x in generate_legal_moves(qval) if x not in scores]
        for v in newvals:
            if scores.get(v, 99999) > scores[qval] + 1:
                scores[v] = scores[qval] + 1
                parents[v] = qval
            totvs += 1
            if v == goal:
                path = [v]
                while path[-1] != start:
                    parent = parents[path[-1]]
                    path.append(parent)
                return path
        for v in newvals:
            heapq.heappush(queue, (distance_score(v), v))
    raise Exception("huh")
