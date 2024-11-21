import math, random

hashpower = [float(x) for x in open('hashpower.csv').readlines()]

# Target block time
TARGET = 12
# Should be 86400, but can reduce for a quicker sim
SECONDS_IN_DAY = 86400
# Look at the 1/x day exponential moving average
EMA_FACTOR = 0.01
# Damping factor for simple difficulty adjustment
SIMPLE_ADJUST_DAMPING_FACTOR = 20
# Maximum per-block diff adjustment (as fraction of current diff)
SIMPLE_ADJUST_MAX = 0.5
# Damping factor for quadratic difficulty adjustment
QUADRATIC_ADJUST_DAMPING_FACTOR = 3
# Maximum per-block diff adjustment (as fraction of current diff)
QUADRATIC_ADJUST_MAX = 0.5
# Threshold for bounded adjustor
BOUNDED_ADJUST_THRESHOLD = 1.3
# Bounded adjustment factor
BOUNDED_ADJUST_FACTOR = 0.01
# How many blocks back to look
BLKS_BACK = 10
# Naive difficulty adjustment factor
NAIVE_ADJUST_FACTOR = 1/1024.


# Produces a value according to the exponential distribution; used
# to determine the time until the next block given an average block
# time of t
def expdiff(t):
    return -math.log(random.random()) * t


# abs_sqr(3) = 9, abs_sqr(-7) = -49, etc
def abs_sqr(x):
    return -(x**2) if x < 0 else x**2


# Given an array of the most recent timestamps, and the most recent
# difficulties, compute the next difficulty
def simple_adjust(timestamps, diffs):
    if len(timestamps) < BLKS_BACK + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-BLKS_BACK] + 0.0
    # Expected interval
    expected = TARGET * BLKS_BACK
    # Compute adjustment factor
    fac = 1 - (delta / expected - 1) / SIMPLE_ADJUST_DAMPING_FACTOR
    fac = max(min(fac, 1 + SIMPLE_ADJUST_MAX), 1 - SIMPLE_ADJUST_MAX)
    return diffs[-1] * fac


# Alternative adjustment algorithm
def quadratic_adjust(timestamps, diffs):
    if len(timestamps) < BLKS_BACK + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-BLKS_BACK] + 0.0
    # Expected interval
    expected = TARGET * BLKS_BACK
    # Compute adjustment factor
    fac = 1 - abs_sqr(delta / expected - 1) / QUADRATIC_ADJUST_DAMPING_FACTOR
    fac = max(min(fac, 1 + QUADRATIC_ADJUST_MAX), 1 - QUADRATIC_ADJUST_MAX)
    return diffs[-1] * fac


# Alternative adjustment algorithm
def bounded_adjust(timestamps, diffs):
    if len(timestamps) < BLKS_BACK + 2:
        return diffs[-1]
    # Total interval between previous block and block a bit further back
    delta = timestamps[-2] - timestamps[-2-BLKS_BACK] + 0.0
    # Expected interval
    expected = TARGET * BLKS_BACK
    if delta / expected > BOUNDED_ADJUST_THRESHOLD:
        fac = (1 - BOUNDED_ADJUST_FACTOR)
    elif delta / expected < 1 / BOUNDED_ADJUST_THRESHOLD:
        fac = (1 + BOUNDED_ADJUST_FACTOR) ** (delta / expected)
    else:
        fac = 1
    return diffs[-1] * fac


# Old Ethereum algorithm
def old_adjust(timestamps, diffs):
    if len(timestamps) < 2:
        return diffs[-1]
    delta = timestamps[-1] - timestamps[-2]
    expected = TARGET * 0.693
    if delta > expected:
        fac = 1 - NAIVE_ADJUST_FACTOR
    else:
        fac = 1 + NAIVE_ADJUST_FACTOR
    return diffs[-1] * fac


def test(source, adjust):
    # Variables to keep track of for stats purposes
    ema = maxema = minema = TARGET
    lthalf, gtdouble, lttq, gtft = 0, 0, 0, 0
    count = 0
    # Block times
    times = [0]
    # Block difficulty values
    diffs = [source[0]]
    # Next time to print status update
    nextprint = 10**6
    # Main loop
    while times[-1] < len(source) * SECONDS_IN_DAY:
        # Print status update every 10**6 seconds
        if times[-1] > nextprint:
            print '%d out of %d processed, ema %f' % \
                (times[-1], len(source) * SECONDS_IN_DAY, ema)
            nextprint += 10**6
        # Grab hashpower from data source
        hashpower = source[int(times[-1] // SECONDS_IN_DAY)]
        # Calculate new difficulty
        diffs.append(adjust(times, diffs))
        # Calculate next block time
        times.append(times[-1] + expdiff(diffs[-1] / hashpower))
        # Calculate min and max ema
        ema = ema * (1 - EMA_FACTOR) + (times[-1] - times[-2]) * EMA_FACTOR
        minema = min(minema, ema)
        maxema = max(maxema, ema)
        count += 1
        # Keep track of number of blocks we are below 75/50% or above
        # 133/200% of target
        if ema < TARGET * 0.75:
            lttq += 1
            if ema < TARGET * 0.5:
                lthalf += 1
        elif ema > TARGET * 1.33333:
            gtft += 1
            if ema > TARGET * 2:
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

# Example usage
# blkdiff.test(blkdiff.hashpower, blkdiff.simple_adjust)
