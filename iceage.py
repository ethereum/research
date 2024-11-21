import random
import datetime

diffs = [3005 * 10**12]
hashpower = diffs[0] / 14
times = [1541247118]


for i in range(6635692, 13000000):
    blocktime = random.expovariate(hashpower / diffs[-1])
    adjfac = max(1 - int(blocktime / 10), -99) / 2048.
    newdiff = diffs[-1] * (1 + adjfac)
    period = (i - 200000) // 100000 - 32
    if i > 200000:
        newdiff += 2 ** period
    diffs.append(newdiff)
    times.append(times[-1] + blocktime)
    if i % 10000 == 0:
        print('Block %d, approx ETH supply %d, time %r blocktime %.2f' % \
            (i, 60102216 * 1.199 + 5.3 * i, datetime.datetime.utcfromtimestamp(times[-1]).isoformat().replace('T',' '), diffs[-1] / hashpower))
        # print int(adjfac * 2048)
