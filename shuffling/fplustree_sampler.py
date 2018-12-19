# Fast sampler without replacement
# Speed
# initialisation - O(n)
# sampling and update - O(log n)
#
# Memory - O(2n)

# Based on paper - A New Data Structure for Cumulative Frequency Tables
#       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.8917&rep=rep1&type=pdf
# And - A Scalable Asynchronous Distributed Algorithm for Topic Modeling
#       https://www.cs.utexas.edu/~rofuyu/papers/nomad-lda-www.pdf

# The main use case of this data structure if for sampling without replacement for
# unbalanced weights or probability distribution for example:
#   - [0.1, 0.4, 0.3, 0.2]
#   - [2, 5, 10, 1]
# It is used at scale to sample from language models of up to billion words
# We can also try it on balance weights.

from hashlib import blake2s
def hash(x): return blake2s(x).digest()[:32]

MAX_U32 = 2**32 - 1

def nextPowerOf2(n):
    return 1 << (n.bit_length() - 1)

class Sampler():
    ## Randomly samples from a list of indices without replacement
    ## The sampler is a binary tree stored in a list.
    ## For a list of size n, n being a power of 2
    ##   - the leaves from [n-1, 2n-1[ contains the probability (if sum to 1)
    ##     or the weight of each item
    ##   - [0, n-1[ contains the internal nodes. They store the cumulative probability of their children
    ##   - The children of a node n are at position 2n+1 and 2n+2
    ##   - The parent of a node n is at position (n-1)/2
    tree = []
    leaves_offset = 0
    seed = []         # 32 bytes
    seed_idx = 0      # Every 4 bytes of the seed is consumed, after 8 times we need to reseed

    def __init__(self, n, seed):
      self.seed = hash(seed)

      ## n, the length of the list you want to sample from
      length = nextPowerOf2(n)
      self.leaves_offset = n-1
      size = self.leaves_offset + n

      # Initialise a list of length 2n+1 since for Ethereum
      # we have a uniform distribution, we can use 1 for all the weights
      self.tree = [1] * size

      # Now build the internal nodes. We iterate in reverse
      for i in range(self.leaves_offset-1, -1, -1): # leaves_offset-1 --> 0 by steps of -1
          self.tree[i] = self.tree[2*i+1] + self.tree[2*i+2]

    def _uniform(self):
        ## Generate an uniform value between [0:self.tree[0][
        ## using a hash seed. Reseed if necessary
        while True:
            candidate = int.from_bytes(self.seed[self.seed_idx:self.seed_idx+4], byteorder='big')

            # Bookkeeping
            self.seed_idx += 4
            if self.seed_idx == 32:
                self.seed = hash(self.seed)
                self.seed_idx -= 32

            if candidate <= MAX_U32 - (MAX_U32 % self.tree[0]):
                return candidate % (self.tree[0])

    def _sample_impl(self, u):
        ## u is a number between 0 and self.tree[0] (the cumulated weight)
        ## Returns the index sampled in the tree.
        ## Leaves_offset must be substracted for real position.
        i = 0

        # A F+tree guarantees the following:
        #   if u >= left CDF (lCDF) => result ∈ right branch
        #   and left branch otherwise
        while i < self.leaves_offset:
            left = 2*i+1
            pLeft = self.tree[left]
            if u >= pLeft:
                # We choose the right child and substract the left CDF (Cumulated Distribution Function)
                # to maintain u ∈ [0, right CDF]
                u -= pLeft
                i = left + 1
            else:
                i = left
        return i

    def sample(self):
        ## Sample from the sampler
        ## Index sampled is **NOT** removed from the indices pool
        ## Only log2(2n-1) comparisons are necessary
        assert self.tree[0] != 0 # assert not empty
        u = self.__uniform()
        return self.__sample_impl(u) - self.leaves_offset

    def _remove_impl(self, idx):
      pos = idx
      self.tree[pos] = 0 # weight = 0 --> no more selected.
      while pos > 0:
          # Propagate the change in cumulative distribution
          # Only log2(2n-1) changes are necessary
          pos = (pos - 1) >> 1 # Jump to parent at (n - 1)/2, rounding if odd is intended.
          self.tree[pos] = self.tree[2*pos+1] + self.tree[2*pos+2]

    def sample_and_remove(self):
        assert self.tree[0] != 0 # assert not empty
        u = self._uniform()
        pos = self._sample_impl(u)
        self._remove_impl(pos)
        return pos - self.leaves_offset

    def sample_and_remove_multi(self, n):
        assert self.tree[0] >= n # assert enough elements
        result = []
        for i in range(n):
            u = self._uniform()
            pos = self._sample_impl(u)
            self._remove_impl(pos)
            result.append(pos - self.leaves_offset)
        return result

if __name__ == '__main__':
    def check_uniform():
        # Sanity checks on uniform distribution with replacement
        n = 100
        sampler = Sampler(n, hash(b'doge'*8))

        freqs = {i:0 for i in range(sampler.tree[0])}
        for i in range(100000):
          freqs[sampler._uniform()] += 1
        print(freqs)
    # check_uniform()

    def check_sampling():
        sampler = Sampler(100000, hash(b'doge'*8))
        committee = sampler.sample_and_remove_multi(500)
        print(committee)
    check_sampling()
