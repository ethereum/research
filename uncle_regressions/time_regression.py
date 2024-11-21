import sys
data = [[float(y) for y in x.strip().split(', ')] for x in open('gethtime.csv').readlines()]
data2 = [[float(y) for y in x.strip().split(', ')] for x in open('old_gethtime.csv').readlines()]

for i in range(0, 2283416, 200000):
    print 'Checking 200k blocks from %d' % i
    dataset = []
    dataset2 = []
    for (d, ds, name) in [(data, dataset, 'old'), (data2, dataset2, 'new')]:
        for j in range(i, min(i + 200000, 2283400), 100):
            gas = 0
            txs = 0
            uncles = 0
            time = 0
            for num, _txs, _gas, _uncles, _time in d[j:j+100]:
                gas += _gas
                txs += _txs
                uncles += _uncles
                time += _time
            ds.append([gas, txs, uncles, time])

        mean_x = sum([x[0] for x in ds]) * 1.0 / len(ds)
        mean_y = sum([x[-1] for x in ds]) * 1.0 / len(ds)
        covar = sum([(x[0] - mean_x) * (x[-1] - mean_y) for x in ds])
        var = sum([(x[0] - mean_x) ** 2 for x in ds])
    
        print name+':'
        print 'm = ', covar / var, '(microseconds per gas)'
        print 'b = ', mean_y - mean_x * (covar / var), '(per 100 blocks)'
