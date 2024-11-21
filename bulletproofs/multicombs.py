import random, time, sys, math

def multisubset(numbers, subsets, adder=lambda x,y: x+y, zero=0):
    # Split up the numbers into partitions
    partition_size = 1 + int(math.log(len(subsets) + 1))
    # Align number count to partition size (for simplicity)
    numbers = numbers[::]
    while len(numbers) % partition_size != 0:
        numbers.append(zero)
    # Compute power set for each partition (eg. a, b, c -> {0, a, b, a+b, c, a+c, b+c, a+b+c})
    power_sets = []
    for i in range(0, len(numbers), partition_size):
        new_power_set = [zero]
        for dimension, value in enumerate(numbers[i:i+partition_size]):
            new_power_set += [adder(n, value) for n in new_power_set]
        power_sets.append(new_power_set)
    # Compute subset sums, using elements from power set for each range of values
    # ie. with a single power set lookup you can get the sum of _all_ elements in
    # the range partition_size*k...partition_size*(k+1) that are in that subset
    subset_sums = []
    for subset in subsets:
        o = zero
        for i in range(len(power_sets)):
            index_in_power_set = 0
            for j in range(partition_size):
                if i * partition_size + j in subset:
                    index_in_power_set += 2 ** j
            o = adder(o, power_sets[i][index_in_power_set])
        subset_sums.append(o)
    return subset_sums

# Reduces a linear combination `numbers[0] * factors[0] + numbers[1] * factors[1] + ...`
# into a multi-subset problem, and computes the result efficiently
def lincomb(numbers, factors, adder=lambda x,y: x+y, zero=0):
    # Maximum bit length of a number; how many subsets we need to make
    maxbitlen = max(len(bin(f))-2 for f in factors)
    # Compute the subsets: the ith subset contains the numbers whose corresponding factor
    # has a 1 at the ith bit
    subsets = [{i for i in range(len(numbers)) if factors[i] & (1 << j)} for j in range(maxbitlen+1)]
    subset_sums = multisubset(numbers, subsets, adder=adder, zero=zero)
    # For example, suppose a value V has factor 6 (011 in increasing-order binary). Subset 0
    # will not have V, subset 1 will, and subset 2 will. So if we multiply the output of adding
    # subset 0 with twice the output of adding subset 1, with four times the output of adding
    # subset 2, then V will be represented 0 + 2 + 4 = 6 times. This reasoning applies for every
    # value. So `subset_0_sum + 2 * subset_1_sum + 4 * subset_2_sum` gives us the result we want.
    # Here, we compute this as `((subset_2_sum * 2) + subset_1_sum) * 2 + subset_0_sum` for
    # efficiency: an extra `maxbitlen * 2` group operations.
    o = zero
    for i in range(len(subsets)-1, -1, -1):
        o = adder(adder(o, o), subset_sums[i])
    return o

# Tests go here
def make_mock_adder():
    counter = [0]
    def adder(x, y):
        if x and y:
            counter[0] += 1
        return x+y
    return adder, counter

def test_multisubset(numcount, setcount):
    numbers = [random.randrange(10**20) for _ in range(numcount)]
    subsets = [{i for i in range(numcount) if random.randrange(2)} for i in range(setcount)]
    adder, counter = make_mock_adder()
    o = multisubset(numbers, subsets, adder=adder)
    for output, subset in zip(o, subsets):
        assert output == sum([numbers[x] for x in subset])

def test_lincomb(numcount, bitlength=256):
    numbers = [random.randrange(10**20) for _ in range(numcount)]
    factors = [random.randrange(2**bitlength) for _ in range(numcount)]
    adder, counter = make_mock_adder()
    o = lincomb(numbers, factors, adder=adder)
    assert o == sum([n*f for n,f in zip(numbers, factors)])
    total_ones = sum(bin(f).count('1') for f in factors)
    print("Naive operation count: %d" % (bitlength * numcount + total_ones))
    print("Optimized operation count: %d" % (bitlength * 2 + counter[0]))
    print("Optimization factor: %.2f" % ((bitlength * numcount + total_ones) / (bitlength * 2 + counter[0])))

if __name__ == '__main__':
    test_lincomb(int(sys.argv[1]) if len(sys.argv) >= 2 else 80)
