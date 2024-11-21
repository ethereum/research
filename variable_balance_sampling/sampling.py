import random, math

def sample_balance(minval, maxval):
    # Gives a power law distribution, where half the validators will be between
    # minval and minval*2, a quarter will be between minval*2 and minval*4...
    return maxval / (1 + random.random() * (maxval / minval - 1))

def expected_balance(min_balance, max_balance):
    # Use integral(1/x) = ln(x)
    ratio = max_balance / min_balance
    if ratio == 1:
        return max_balance
    return max_balance * math.log(ratio) / (ratio - 1)

def sample_attack(min_balance, max_balance, sample_size, global_bad_share):
    # Returns attacker fraction of the sample
    expected_good_balance = expected_balance(min_balance, max_balance)
    bad_per_good_balance = global_bad_share / (1 - global_bad_share)
    bad_per_good_validators = (
        bad_per_good_balance / max_balance * expected_good_balance
    )
    avg_bad_validator_share = (
        bad_per_good_validators / (1 + bad_per_good_validators)
    )
    good_total = 0
    bad_total = 0
    for i in range(sample_size):
        if random.random() < avg_bad_validator_share:
            bad_total += max_balance
        else:
            good_total += sample_balance(min_balance, max_balance)
    return bad_total / (bad_total + good_total)

def variance(data):
    n=len(list(data))
    return sum(x*x for x in data)/n - (sum(list(data))/n)**2

def standev(data):
    return variance(data)**0.5

def test():
    params = (
        # (min_balance, max_balance, sample_size)
        (32, 32, 1024),
        (32, 256, 1024),
        (32, 2048, 1024),
        (32, 16384, 1024),
        (32, 131072, 1024),
    )

    for (_min, _max, _sam) in params:
        print(f"Testing: min={_min} max={_max} sample_size={_sam}")
        o = [sample_attack(_min, _max, _sam, 1/3) for i in range(10000)]
        print("Attacker share standard deviation: {}".format(standev(o)))
        top_milli = sorted(o)[-10]
        print(f"Attacker top 0.1 percentile share: {top_milli}")

if __name__ == '__main__':
    test()
