import math, random

hashpower = [float(x) for x in open('hashpower.csv').readlines()]

target = 12
seconds_in_day = 86400
ema_factor = 0.01
f = 20
sqrf = 3
threshold = 1.3
adj_factor = 0.01
maxadjust = 0.5
blks_back = 10


def expdiff(t):
    return -math.log(random.random()) * t


def calc_threshold_time(p, t):
    return t * -math.log(1 - p)


def abs_sqr(x):
    return -(x**2) if x < 0 else x**2


def simple_adjust(timestamps, diffs):
    if len(timestamps) < blks_back + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-blks_back] + 0.0
    # Expected interval
    expected = target * blks_back
    fac = max(min(1 - (delta / expected - 1) / f, 1+maxadjust), 1-maxadjust)
    return diffs[-1] * fac


def quadratic_adjust(timestamps, diffs):
    if len(timestamps) < blks_back + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-blks_back] + 0.0
    # Expected interval
    expected = target * blks_back
    fac = max(min(1 - abs_sqr(delta / expected - 1) / sqrf,
                  1+maxadjust), 1-maxadjust)
    return diffs[-1] * fac


def bounded_adjust(timestamps, diffs):
    if len(timestamps) < blks_back + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-blks_back] + 0.0
    # Expected interval
    expected = target * blks_back
    if delta / expected > threshold:
        fac = (1 - adj_factor)
    elif delta / expected < 1 / threshold:
        fac = (1 + adj_factor) ** (delta / expected)
    else:
        fac = 1
    return diffs[-1] * fac


def test(source, adjust):
    ema = maxema = minema = target
    lthalf, gtdouble, lttq, gtft = 0, 0, 0, 0
    times = [0]
    diffs = [source[0]]
    nextprint = 10**6
    count = 0
    while times[-1] < len(source) * seconds_in_day:
        if times[-1] > nextprint:
            print '%d out of %d processed' % \
                (times[-1], len(source) * seconds_in_day)
            nextprint += 10**6
        # Grab hashpower from data source
        hashpower = source[int(times[-1] // seconds_in_day)]
        # Calculate new difficulty
        diffs.append(adjust(times, diffs))
        # Calculate next block time
        times.append(times[-1] + expdiff(diffs[-1] / hashpower))
        # Calculate min and max ema
        ema = ema * (1 - ema_factor) + (times[-1] - times[-2]) * ema_factor
        minema = min(minema, ema)
        maxema = max(maxema, ema)
        count += 1
        if ema < target * 0.75:
            lttq += 1
            if ema < target * 0.5:
                lthalf += 1
        elif ema > target * 1.33333:
            gtft += 1
            if ema > target * 2:
                gtdouble += 1
        # Pop items to save memory
        if len(times) > 2000:
            times.pop(0)
            diffs.pop(0)
    print 'min', minema, 'max', maxema, 'avg', times[-1] / count, \
        'ema < half', lthalf * 1.0 / count, \
        'ema > double', gtdouble * 1.0 / count, \
        'ema < 3/4', lttq * 1.0 / count, \
        'ema > 4/3', gtft * 1.0 / count
