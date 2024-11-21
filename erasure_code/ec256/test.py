import random
import share

fsz = 200

f = ''.join([
    random.choice('1234567890qwetyuiopasdfghjklzxcvbnm') for x in range(fsz)])

c = share.split_file(f, 5, 4)

print 'File split successfully.'
print ' '
print 'Chunks: '
print ' '
for chunk in c:
    print chunk
print ' '

g = ''.join([
    random.choice('1234567890qwetyuiopasdfghjklzxcvbnm') for x in range(fsz)])

c2 = share.split_file(g, 5, 4)

assert share.recombine_file(
    [c[0], c2[1], c[2], c[3], c2[4], c[5], c[6], c[7], c[8]]) == f

print '5 of 9 with 7 legit, 2 errors passed'

assert share.recombine_file(
    [c[0], c[2], c[3], c2[4], c[6], c[7], c[8]]) == f

print '5 of 9 with 6 legit, 1 error passed'

assert share.recombine_file(
    [c[0], c[3], c[6], c[7], c[8]]) == f

print '5 of 9 with 5 legit, 0 errors passed'

chunks3 = share.serialize_cubify(f, 3, 3, 2)

print ' '
print 'Chunks: '
print ' '
for chunk in chunks3:
    print chunk
print ' '

for i in range(26):
    pos = random.randrange(len(chunks3))
    print 'Removing cell %d' % pos
    chunks3.pop(pos)

assert share.full_heal_set(chunks3) == f

print ' '
print 'Cube reconstruction test passed'
print ' '

chunks4 = share.serialize_cubify(g, 3, 3, 2)

for i in range(7):
    pos = random.randrange(len(chunks3))
    chunk = chunks3.pop(pos)
    print 'Damaging cell %d' % pos
    print 'Prior: %s' % chunk
    metadata, content = share.deserialize_chunk(chunk)
    for j in range(len(content)):
        content[j] = random.randrange(256)
    chunks3.append(share.serialize_chunk(content, *metadata))
    print 'Post: %s' % chunks3[-1]

assert share.full_heal_set(chunks4) == g

print ' '
print 'Byzantine cube reconstruction test passed'
