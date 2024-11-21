import spread
import math
import random

o = spread.declutter(spread.load('diff_txs_price.csv'))

diffs = [float(q[2]) for q in o]
prices = [float(q[1]) for q in o]
txs = [float(q[3]) for q in o]
txfees = [float(q[4]) for q in o]


def simple_estimator(fac):
    o = [1]
    for i in range(1, len(diffs)):
        o.append(o[-1] * diffs[i] * 1.0 / diffs[i-1] / fac)
    return o


def minimax_estimator(fac):
    o = [1]
    for i in range(1, len(diffs)):
        if diffs[i] * 1.0 / diffs[i-1] > fac:
            o.append(o[-1] * diffs[i] * 1.0 / diffs[i-1] / fac)
        elif diffs[i] > diffs[i-1]:
            o.append(o[-1])
        else:
            o.append(o[-1] * diffs[i] * 1.0 / diffs[i-1])
    return o


def diff_estimator(fac, dw, mf, exp=1):
    o = [1]
    derivs = [0] * 14
    for i in range(14, len(diffs)):
        derivs.append(diffs[i] - diffs[i - 14])
    for i in range(0, 14):
        derivs[i] = derivs[14]
    vals = [max(diffs[i] + derivs[i] * dw, diffs[i] * mf) for i in range(len(diffs))]
    for i in range(1, len(diffs)):
        if vals[i] * 1.0 / vals[i-1] > fac:
            o.append(o[-1] * 1.0 / fac * (vals[i] / vals[i-1])**exp)
        elif vals[i] > vals[i-1]:
            o.append(o[-1])
        else:
            o.append(o[-1] * 1.0 * (vals[i] / vals[i-1])**exp)
    return o


def tx_diff_estimator(fac, dw, mf, lin=1, exp=1):
    fac = (fac - 1) or 0.000001
    o = [1]
    initavg = sum([txs[i] for i in range(5)]) / 5.0
    txavgs = [initavg] * 5
    for i in range(5, len(txs)):
        txavgs.append(txavgs[-1] * 0.8 + txs[i] * 0.2)

    derivs = [0] * 14
    for i in range(14, len(txavgs)):
        derivs.append(txavgs[i] - txavgs[i - 14])
    for i in range(0, 14):
        derivs[i] = derivs[14]
    vals = [max(txavgs[i] + derivs[i] * dw, txavgs[i] * mf) for i in range(len(txavgs))]
    for i in range(1, len(txavgs)):
        growth = (vals[i] * 1.0 / vals[i-1] - 1)
        if growth > fac:
            surplus = (growth / fac) - 1
            o.append(o[-1] * (1 + (surplus * lin * fac) ** exp))
        elif vals[i] > vals[i-1]:
            o.append(o[-1])
        else:
            surplus = 1 - growth
            o.append(o[-1] * (1 - (surplus * lin * fac) ** exp))
        if i and o[-1] < o[-2] * mf:
            o[-1] = o[-2] * mf
    return o


def minimax_fee_estimator(fac, days):
    o = [1]
    initavg = sum([txs[i] for i in range(int(days))]) * 1.0 / days
    txavgs = [initavg] * int(days)
    for i in range(int(days), len(txs)):
        txavgs.append(txavgs[-1] * 1.0 * (days-1) / days + txs[i] * 1.0 / days)
    initavg2 = sum([txfees[i] for i in range(int(days))]) * 1.0 / days
    txfeeavgs = [initavg2] * int(days)
    for i in range(int(days), len(txs)):
        txfeeavgs.append(txfeeavgs[-1] * 1.0 * (days-1) / days + txfees[i] * 1.0 / days)
    # Calculate inverse fee, invfee ~= price
    txavgfees = [t / f for f, t in zip(txfeeavgs, txavgs)]
    for i in range(1, len(txavgfees)):
        if txavgfees[i] * 1.0 / txavgfees[i-1] > fac:
            o.append(o[-1] * txavgfees[i] * 1.0 / txavgfees[i-1] / fac)
        elif txavgfees[i] > txavgfees[i-1]:
            o.append(o[-1])
        else:
            o.append(o[-1] * txavgfees[i] * 1.0 / txavgfees[i-1])
    return o


def ndiff_estimator(*args):
    fac, dws, mf = args[0], args[1:-1], args[-1]
    o = [1]
    ds = [diffs]
    for dw in dws:
        derivs = [0] * 14
        for i in range(14, len(diffs)):
            derivs.append(ds[-1][i] - ds[-1][i - 14])
        for i in range(0, 14):
            derivs[i] = derivs[14]
        ds.append(derivs)
    vals = []
    for i in range(len(diffs)):
        q = ds[0][i] + sum([ds[j+1][i] * dws[j] for j in range(len(dws))])
        vals.append(max(q, ds[0][i] * mf))
    for i in range(1, len(diffs)):
        if vals[i] * 1.0 / vals[i-1] > fac:
            o.append(o[-1] * vals[i] * 1.0 / vals[i-1] / fac)
        elif vals[i] > vals[i-1]:
            o.append(o[-1])
        else:
            o.append(o[-1] * vals[i] * 1.0 / vals[i-1])
    return o


def dual_threshold_estimator(fac1, fac2, dmul):
    o = [1]
    derivs = [0] * 14
    for i in range(14, len(diffs)):
        derivs.append(diffs[i] - diffs[i - 14])
    for i in range(0, 14):
        derivs[i] = derivs[14]
    for i in range(1, len(diffs)):
        if diffs[i] * 1.0 / diffs[i-1] > fac1 and derivs[i] * 1.0 / derivs[i-1] > fac2:
            o.append(o[-1] * diffs[i] * 1.0 / diffs[i-1] / fac1 * (1 + (derivs[i] / derivs[i-1] - fac2) * dmul))
        elif diffs[i] > diffs[i-1]:
            o.append(o[-1])
        else:
            o.append(o[-1] * diffs[i] * 1.0 / diffs[i-1])
    return o

infinity = 2.**1023
infinity *= 2


def evaluate_estimates(estimates, crossvalidate=False):
    sz = len(prices) if crossvalidate else 780
    sqdiffsum = 0
    # compute average
    tot = 0
    for i in range(sz):
        if estimates[i] == infinity or estimates[i] <= 0:
            return 10**20
        tot += math.log(prices[i] / estimates[i])
    avg = 2.718281828459 ** (tot * 1.0 / sz)
    if avg <= 0:
        return 10**20
    for i in range(1, sz):
        sqdiffsum += math.log(prices[i] / estimates[i] / avg) ** 2
    return sqdiffsum


# Simulated annealing optimizer
def optimize(producer, floors, ceilings, rate=0.7, rounds=5000, tries=1):
    bestvals, besty = None, 10**21
    for t in range(tries):
        print 'Starting test %d of %d' % (t + 1, tries)
        vals = [f*0.5+c*0.5 for f, c in zip(floors, ceilings)]
        y = evaluate_estimates(producer(*vals))
        for i in range(1, rounds):
            stepsizes = [(f*0.5-c*0.5) / i**rate for f, c in zip(floors, ceilings)]
            steps = [(random.random() * 2 - 1) * s for s in stepsizes]
            newvals = [max(mi, min(ma, v+s)) for v, s, mi, ma in zip(vals, steps, floors, ceilings)]
            newy = evaluate_estimates(producer(*newvals))
            if newy < y:
                vals = newvals
                y = newy
            if not i % 1000:
                print i, vals, y
        if y < besty:
            bestvals, besty = vals, y
        
    return bestvals


def score(producer, *vals):
    return evaluate_estimates(producer(*vals), True)
