import math
import random

hashpower = [float(x) for x in open('hashpower.csv').readlines()]

target = 12
seconds_in_day = 8640
ema_factor = 0.005
f = 30
threshold = 1.5
maxadjust = 0.1


def expdiff(t):
    return -math.log(random.random()) * t


def adjust(timestamps, diffs):
    if len(timestamps) < 7:
        return diffs[-1]
    blks_back = 5
    delta = timestamps[-2] - timestamps[-2-blks_back]
    expected = target * 0.693 * blks_back
    if delta < expected / threshold:
        fac = max(1 + 1. * delta / expected / f, 1 + maxadjust) ** 2
    elif delta > expected * threshold:
        fac = max(1 - 1. * delta / expected / f, 1 - maxadjust) ** 2
    else:
        fac = 1
    return diffs[-1] * fac


def test(source, adjust):
    ema = target
    chain = [0]
    d = [source[0]]
    emachain = [target] 
    sourcechain = [0]
    while chain[-1] < len(source) * seconds_in_day:
        hashpower = source[int(chain[-1] // seconds_in_day)]
        d.append(adjust(chain, d))
        chain.append(chain[-1] + expdiff(d[-1] / hashpower))
        ema = ema * (1 - ema_factor) + (chain[-1] - chain[-2]) * ema_factor
        emachain.append(ema)
        sourcechain.append(hashpower)
    print 'min', min(emachain[500:]), 'max', max(emachain[500:]), 'avg', sum(emachain) / len(emachain)
    return zip(emachain[500:], sourcechain[500:], chain[500:])
