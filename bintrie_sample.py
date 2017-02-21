import bintrie

datapoints = [1, 3, 10, 31, 100, 316, 1000, 3162]
o = []

for i in range(len(datapoints)):
    p = []
    for j in range(i+1):
        print 'Running with: %d %d' % (datapoints[i], datapoints[j])
        p.append(bintrie.test(datapoints[i], datapoints[j])['compressed_db_size'])
    o.append(p)

print o
