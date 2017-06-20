import numpy, random

EXP_CONN = 2
NODES = 10000

nodes = []

for i in range(NODES):
    n = []
    EDGES = numpy.random.poisson(EXP_CONN / 2.)
    for j in range(EDGES):
        n.append(random.randrange(NODES))
    nodes.append(n)

for i in range(NODES):
    for p in nodes[i]:
        if i not in nodes[p]:
            nodes[p].append(i)

print 'Total edges:', sum([len(s) for s in nodes])

components = []
scanned = {}
for i in range(NODES):
    if i in scanned:
        pass
    size = 0
    positions = [i]
    while len(positions):
        val = positions.pop()
        if val not in scanned:
            scanned[val] = True
            size += 1
            for p in nodes[val]:
                positions.append(p)
    # print 'Component:', size
    components.append(size)

print 'Biggest components:', sorted(components)[::-1][:10]
