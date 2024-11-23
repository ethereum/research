# Checking different implementations for a 'compression dictionary' that can
# dynamically update based on which values are the most often used. The idea
# is that we maintain an on-chain size-192 dictionary, which stores the
# most-often-used addresses, and every time a transaction contains one of
# those addresses, that address can be replaced by a single byte representing
# its position in the dictionary.
#
# There is value in having a dictionary auto-update on-chain, because this
# would make it decentralized infrastructure and credibly neutral

DICT_SIZE = 192
VOCAB_SIZE = 100000
SAMPLE_COUNT = 10000
import random
import math

# Returns 0....maximum-1, where prob(i) is proportional to 1/(i+1)
# This is a common probability distribution in statistics, see
# https://en.wikipedia.org/wiki/Zipf%27s_law
def zipfs_law_sample(maximum):
    return int(math.exp(random.random() * math.log(maximum + 1))) - 1

# Evaluates a given algorithm for updating the dictionary
def evaluate_dict_algorithm(Algo, trials=1, trace_interval=0):
    hits_trace = []
    total_hits = 0
    for trial in range(trials):
        algo = Algo()
        hits = 0
        for sample in range(SAMPLE_COUNT):
            v = zipfs_law_sample(VOCAB_SIZE)
            if algo.in_dict(v):
                hits += 1
            if trace_interval !=0 and sample % trace_interval == 0:
                if trial == 0:
                    hits_trace.append(hits)
                else:
                    hits_trace[sample // trace_interval] += hits
            algo.process(v)
        total_hits += hits
    if trace_interval == 0:
        return total_hits
    else:
        return total_hits, hits_trace

# Stores the frequency table. The dictionary is the top N addresses in
# the frequency table. This is expensive!
class AlgoTopn():
    def __init__(self):
        self.freqs = {}

    def in_dict(self, v):
        ordered = sorted(
            list(self.freqs.keys()),
            key = lambda x: -self.freqs[x]
        )
        return v in ordered[:DICT_SIZE]

    def process(self, v):
        self.freqs[v] = self.freqs.get(v, 0) + 1

# Stores the frequency table. The dictionary is the top N addresses in
# the frequency table using a heap (with sift limit).
class AlgoHeap():
    def __init__(self, sift_limit=64):
        self.freqs = {}
        self.dict = {}       # index to value mapping
        self.positions = {}  # value to index mapping
        self.sift_limit = sift_limit

    def in_dict(self, v):
        return v in self.dict.values()

    def siftDown(self, pos):
        v = self.dict[pos]
        left = pos * 2
        if left >= len(self.dict):
            return None
        left_v = self.dict[left]
        right = pos * 2 + 1
        if right >= len(self.dict) or self.freqs[left_v] < self.freqs[self.dict[right]]:
            min_child = left
            min_v = left_v
        else:
            min_child = right
            min_v = self.dict[right]

        if self.freqs[min_v] < self.freqs[v]:
            # swap
            self.positions[min_v] = pos
            self.dict[pos] = min_v
            self.positions[v] = min_child
            self.dict[min_child] = v
            return min_child
        else:
            return None
    
    def siftUp(self, pos):
        if pos == 0:
            return None
        v = self.dict[pos]
        parent = pos // 2
        parent_v = self.dict[parent]

        if self.freqs[parent_v] > self.freqs[v]:
            # swap
            self.positions[parent_v] = pos
            self.dict[pos] = parent_v
            self.positions[v] = parent
            self.dict[parent] = v
            return parent
        else:
            return None

    def process(self, v):
        self.freqs[v] = self.freqs.get(v, 0) + 1
        if v in self.positions:
            p = self.positions[v]
            while p is not None:
                p = self.siftDown(p)
        elif len(self.dict) < DICT_SIZE:
            self.dict[len(self.dict)] = v
            self.positions[v] = len(self.dict) - 1
            p = len(self.dict) - 1
            while p is not None:
                p = self.siftUp(p)
        elif self.freqs[v] > self.freqs[self.dict[0]]:
            last_value = self.dict[0]
            del self.positions[last_value]
            self.positions[v] = 0
            self.dict[0] = v
            p = 0
            while p is not None:
                p = self.siftDown(p)

# When a new value comes in, it randomly replaces an existing one
class AlgoRandomReplace():
    def __init__(self):
        self.dict = []

    def in_dict(self, v):
        return v in self.dict

    def process(self, v):
        if len(self.dict) < DICT_SIZE:
            self.dict.append(v)
        else:
            self.dict[random.randrange(DICT_SIZE)] = v

# When a new value comes in, it randomly replaces an existing one
# only if it's more frequent
class AlgoFreqReplace():
    def __init__(self):
        self.dict = []
        self.freqs = {}
        self.positions = {}

    def in_dict(self, v):
        return v in self.dict

    def process(self, v):
        self.freqs[v] = self.freqs.get(v, 0) + 1
        if v in self.positions:
            pass
        elif len(self.dict) < DICT_SIZE:
            self.dict.append(v)
            self.positions[v] = len(self.dict) - 1
        else:
            random_index = random.randrange(DICT_SIZE)
            existing_value = self.dict[random_index]
            if self.freqs[v] > self.freqs[existing_value]:
                del self.positions[existing_value]
                self.positions[v] = random_index
                self.dict[random_index] = v

if __name__ == '__main__':
    print("Testing 'Top N' algorithm")
    hits = evaluate_dict_algorithm(AlgoTopn)
    print("{} of {} hits".format(hits, SAMPLE_COUNT))
    print("Testing 'random replace' algorithm")
    hits = evaluate_dict_algorithm(AlgoRandomReplace)
    print("{} of {} hits".format(hits, SAMPLE_COUNT))
    print("Testing 'replace-by-frequency' algorithm")
    hits = evaluate_dict_algorithm(AlgoFreqReplace)
    print("{} of {} hits".format(hits, SAMPLE_COUNT))
    print("Testing 'heap' algorithm")
    hits = evaluate_dict_algorithm(AlgoHeap)
    print("{} of {} hits".format(hits, SAMPLE_COUNT))

    trials = 100 # Monte-Carlo trials
    trace_interval = 10
    print("Large testing with trials = {}".format(trials))
    # print("Testing 'Top N' algorithm")
    # hits = evaluate_dict_algorithm(AlgoTopn, trials)
    # print("{} of {} hits".format(hits, trials * SAMPLE_COUNT))
    # print("Testing 'random replace' algorithm")
    # hits = evaluate_dict_algorithm(AlgoRandomReplace, trials)
    # print("{} of {} hits".format(hits, trials * SAMPLE_COUNT))
    print("Testing 'replace-by-frequency' algorithm")
    hits, trace_rf = evaluate_dict_algorithm(AlgoFreqReplace, trials, trace_interval)
    print("{} of {} hits".format(hits, trials * SAMPLE_COUNT))
    print("Testing 'heap' algorithm")
    hits, trace_heap = evaluate_dict_algorithm(AlgoHeap, trials, trace_interval)
    print("{} of {} hits".format(hits, trials * SAMPLE_COUNT))

    import matplotlib.pyplot as plt
    plt.plot(range(0, len(trace_rf) * trace_interval, trace_interval), [x / trials / (i + 1) / trace_interval for i, x in enumerate(trace_rf)], label="Replace-by-frequency")
    plt.plot(range(0, len(trace_rf) * trace_interval, trace_interval), [x / trials / (i + 1) / trace_interval for i, x in enumerate(trace_heap)], label="Optimal")
    plt.xlabel("Sample")
    plt.ylabel("Hit Rate")
    plt.legend()
    plt.show()