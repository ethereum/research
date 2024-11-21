import sys
data = [[float(y) for y in x.strip().split(', ')] for x in open('attack_gas.csv').readlines()]
# first = int(data[0][0])
first = 2405000
last = int(data[-1][0])

dataset = []
slicenums = [0] * 100
slicedens = [0] * 100
slice_width = 50000
pos = first
i = 0
while data[i][0] < pos:
    i += 1
while i < len(data):
    nxt = pos + 100
    gas = 0
    attackgas = 0
    uncles = 0
    while pos < nxt and i < len(data):
        pos, _gas, _attackgas, _uncles = data[i]
        gas += _gas
        attackgas += _attackgas
        uncles += _uncles
        i += 1
    totblocks = pos - nxt + 100.0
    if totblocks:
        dataset.append([gas / totblocks, attackgas / totblocks, uncles / totblocks])
        slicenums[int(attackgas / totblocks / slice_width)] += uncles / totblocks
        slicedens[int(attackgas / totblocks / slice_width)] += 1.0

mean_x = sum([x[1] for x in dataset]) * 1.0 / len(dataset)
mean_x0 = sum([x[0] for x in dataset]) * 1.0 / len(dataset)
mean_y = sum([x[-1] for x in dataset]) * 1.0 / len(dataset)
covar = sum([(x[1] - mean_x) * (x[-1] - mean_y) for x in dataset])
covar0 = sum([(x[0] - mean_x) * (x[-1] - mean_y) for x in dataset])
var = sum([(x[1] - mean_x) ** 2 for x in dataset])
var0 = sum([(x[0] - mean_x) ** 2 for x in dataset])

print 'm = ', covar / var, '(uncle rate per attack gas)'
print 'b = ', mean_y - mean_x * (covar / var)

print 'm = ', covar / var0, '(uncle rate per gas)'
print 'b = ', mean_y - mean_x0 * (covar / var0)

# print [slicenums[i] / slicedens[i] if slicedens[i] else None for i in range(len(slicenums))]
