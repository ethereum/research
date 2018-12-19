from prime_shuffle import prime_shuffle, prime_shuffle_partial
from feistel_shuffle import feistel_shuffle, feistel_shuffle_partial
from fisher_yates_shuffle import fisher_yates_shuffle
from fplustree_sampler import Sampler
import time

count = 100000
subcount = 500

print("Testing prime shuffle")
a = time.time()
o = prime_shuffle(range(count), b'doge'*8)
print(o[:10])
t2 = time.time()
o2 = prime_shuffle_partial(range(count), b'doge' * 8, subcount)
print(o2[:10])
print("Total runtime: ", t2 - a)
print("Runtime to compute committee: ", time.time() - t2)
print("\n")

print("Testing feistel shuffle")
a = time.time()
o = feistel_shuffle(range(count), b'doge'*8)
print(o[:10])
t2 = time.time()
o2 = feistel_shuffle_partial(range(count), b'doge' * 8, subcount)
print(o2[:10])
print("Total runtime: ", t2 - a)
print("Runtime to compute committee: ", time.time() - t2)
print("\n")

print("Testing Fisher-Yates shuffle")
a = time.time()
o = fisher_yates_shuffle(range(count), b'doge'*8)
print(o[:10])
print("Total runtime: ", time.time() - a)
print("\n")

print("Testing F+tree sampling")
a = time.time()
sampler = Sampler(count, b'doge'*8)
o2 = sampler.sample_and_remove_multi(count)
print(o[:10])
t2 = time.time()
sampler = Sampler(count, b'doge'*8)
o2 = sampler.sample_and_remove_multi(subcount)
print(o2[:10])
t3 = time.time()
o3 = sampler.sample_and_remove_multi(subcount)
print(o3[:10])
print("Total runtime: ", t2 - a)
print("Runtime to compute first committee: ", t3 - t2)
print("Runtime to compute next committee: ", time.time() - t3)
print("\n")
