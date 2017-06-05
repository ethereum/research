data = [[float(y) for y in x.strip().split(', ')] for x in open('block_datadump.csv').readlines()]

for i in range(0, 2283416, 200000):
    print('Checking 200k blocks from %d' % i)
    dataset = []
    for j in range(i, min(i + 200000, 2283400), 100):
        gas = 0
        nonzeroes = 0
        txs = 0
        uncs = 0
        zeroes = 0
        for num, _uncs, uncrew, uncgas, _txs, _gas, _length, _zeroes in data[j:j+100]:
            txs += _txs
            gas += _gas
            nonzeroes += _length - _zeroes
            zeroes += _zeroes
            uncs += _uncs
        dataset.append([gas, txs, nonzeroes, zeroes, uncs])

    mean_x = sum([x[0] for x in dataset]) * 1.0 / len(dataset)
    mean_y = sum([x[-1] for x in dataset]) * 1.0 / len(dataset)
    covar = sum([(x[0] - mean_x) * (x[-1] - mean_y) for x in dataset])
    var = sum([(x[0] - mean_x) ** 2 for x in dataset])

    for d in dataset:
        d.append(d[-1] - covar / var * (d[0] - mean_x))

    mean_x1 = sum([x[1] for x in dataset]) * 1.0 / len(dataset)
    mean_x2 = sum([x[2] for x in dataset]) * 1.0 / len(dataset)
    mean_x3 = sum([x[3] for x in dataset]) * 1.0 / len(dataset)
    mean_y2 = sum([x[-1] for x in dataset]) * 1.0 / len(dataset)
    covar1 = sum([(x[1] - mean_x1) * (x[-1] - mean_y2) for x in dataset])
    var1 = sum([(x[1] - mean_x1) ** 2 for x in dataset])
    covar2 = sum([(x[2] - mean_x2) * (x[-1] - mean_y2) for x in dataset])
    var2 = sum([(x[2] - mean_x2) ** 2 for x in dataset])
    covar3 = sum([(x[3] - mean_x2) * (x[-1] - mean_y2) for x in dataset])
    var3 = sum([(x[3] - mean_x2) ** 2 for x in dataset])
    print('Base m =', covar / var)
    print('Base b =', mean_y - mean_x * (covar / var))
    print('Excess m for txs=', covar1 / var1)
    print('Excess m for nonzero bytes=', covar2 / var2)
    print('Excess m for zero bytes=', covar3 / var3)
