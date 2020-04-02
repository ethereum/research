# For each subset in `subsets` (provided as a list of indices into `numbers`),
# compute the sum of that subset of `numbers`. More efficient than the naive method.
def multisubset(numbers, subsets, adder=lambda x,y: x+y, zero=0):
    numbers = numbers[::]
    subsets = [{x for x in subset} for subset in subsets]
    for roundcount in range(9999999):
        # Compute counts of every pair of indices in the subset list
        count = {}
        for subset in subsets:
            if subset:
                for x in subset:
                    for y in subset:
                        if y > x:
                            count[(x, y)] = count.get((x, y), 0) + 1

        # Exit condition: all subsets have size 1, no pairs
        if not count:
            return [numbers[list(subset)[0]] if subset else zero for subset in subsets]

        # Determine pairs with highest count. The cutoff parameter [:len(numbers)]
        # determines a tradeoff between group operation count and other forms of overhead
        pairs_by_count = sorted([el for el in count.keys()], key=lambda el: count[el], reverse=True)[:len(numbers)]

        # In each of the highest-count pairs, take the sum of the numbers at those indices,
        # and add the result as a new value, and modify `subsets` to include the new value
        # wherever possible
        used = set()
        for maxx, maxy in pairs_by_count:
            if maxx in used or maxy in used:
                continue
            used.add(maxx)
            used.add(maxy)
            numbers.append(adder(numbers[maxx], numbers[maxy]))
            for subset in subsets:
                if maxx in subset and maxy in subset:
                    subset.remove(maxx)
                    subset.remove(maxy)
                    subset.add(len(numbers)-1)

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
import random

def make_mock_adder():
    counter = [0]
    def adder(x, y):
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
    test_lincomb(80)
