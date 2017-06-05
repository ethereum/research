data = [[float(y) for y in x.strip().split(', ')] for x in open('block_datadump.csv').readlines()]

for i in range(0, 2283416, 200000):
    print('Checking 200k blocks from %d' % i)
    dataset = []
    totuncles, totuncreward = 0, 0
    totbs = [0 for j in range(40)]
    totus = [0 for j in range(40)]
    for num, uncs, uncrew, uncgas, txs, gas, length, zeroes in data[i:i+200000]:
        dataset.append([gas, 0])
        for i in range(int(uncs)):
            dataset.append([uncgas / uncs * 1.0, 1])
        totuncles += uncs
        totuncreward += uncrew
        totus[int(gas / 100000)] += uncs
        totbs[int(gas / 100000)] += 1
    print([totus[j] * 100.0 / (totbs[j] + 0.000000001) for j in range(40)])
    print('Average uncle reward:', totuncreward * 1.0 / totuncles)
    print('Average nephew reward:', totuncles * 5 / 32. / len(dataset))
    
    mean_x = sum([x[0] for x in dataset]) * 1.0 / len(dataset)
    mean_y = sum([x[1] for x in dataset]) * 1.0 / len(dataset)
    print('Average gas used:', mean_x)
    print('Average uncle rate:', mean_y)
    
    covar = sum([(x[0] - mean_x) * (x[1] - mean_y) for x in dataset])
    var = sum([(x[0] - mean_x) ** 2 for x in dataset])
    
    print('m = ', covar / var)
    print('b = ', mean_y - mean_x * (covar / var))
