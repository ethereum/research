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
def evaluate_dict_algorithm(Algo):
    algo = Algo()
    hits = 0
    for _ in range(SAMPLE_COUNT):
        v = zipfs_law_sample(VOCAB_SIZE)
        if algo.in_dict(v):
            hits += 1
        algo.process(v)
    return hits

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
