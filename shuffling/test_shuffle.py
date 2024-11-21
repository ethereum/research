from prime_shuffle import prime_shuffle, prime_shuffle_partial
from feistel_shuffle import feistel_shuffle, feistel_shuffle_partial
from swap_or_not_shuffle import swap_or_not_shuffle, swap_or_not_shuffle_partial
from fisher_yates_shuffle import fisher_yates_shuffle
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

print("Testing swap-or-not shuffle")
a = time.time()
o = swap_or_not_shuffle(range(count), b'doge'*8)
print(o[:10])
t2 = time.time()
o2 = swap_or_not_shuffle_partial(range(count), b'doge' * 8, subcount)
print(o2[:10])
print("Total runtime: ", t2 - a)
print("Runtime to compute committee: ", time.time() - t2)
print("\n")

print("Testing Fisher-Yates shuffle")
a = time.time()
o = fisher_yates_shuffle(range(count), b'doge'*8)
print(o[:10])
print("Total runtime: ", time.time() - a)
