import new_bintrie as t1
import new_bintrie_optimized as t2
import new_bintrie_hex as t3
import time
import binascii

keys = [t1.sha3(bytes([i // 256, i % 256])) for i in range(10000)]

d = t1.EphemDB()
r = t1.new_tree(d)
a = time.time()
for k in keys[:1000]:
    r = t1.update(d, r, k, k)
print("Naive bintree time to update: %.4f" % (time.time() - a))
print("Root: %s" % binascii.hexlify(r))

d = t2.EphemDB()
r = t2.new_tree(d)
a = time.time()
for k in keys[:1000]:
    r = t2.update(d, r, k, k)
print("DB-optimized bintree time to update: %.4f" % (time.time() - a))
print("Root: %s" % binascii.hexlify(r))
print("Writes: %d, reads: %d" % (d.writes, d.reads))
d.reads = 0
for k in keys[:500]:
    assert t2.get(d, r, k) == k
for k in keys[-500:]:
    assert t2.get(d, r, k) == b'\x00' * 32
print("Reads: %d" % d.reads)

d = t3.EphemDB()
r = t3.new_tree(d)
a = time.time()
for k in keys[:1000]:
    r = t3.update(d, r, k, k)
print("DB-optimized bintree time to update: %.4f" % (time.time() - a))
print("Root: %s" % binascii.hexlify(r))
print("Writes: %d, reads: %d" % (d.writes, d.reads))
d.reads = 0
for k in keys[:500]:
    assert t3.get(d, r, k) == k
for k in keys[-500:]:
    assert t3.get(d, r, k) == b'\x00' * 32
print("Reads: %d" % d.reads)
