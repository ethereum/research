lines = open('data.csv').read().split('\n')
data = [(int(x[:x.find(',')]), float(x[x.find(',')+1:])) for x in lines if x]

REPORT_THRESHOLD = 0.23

def get_error(scale, elasticity, growth):
    err = 0    
    bs_fac, fee_fac = 1 / (1 + elasticity), elasticity / (1 + elasticity)
    for i, (block_size, avg_fee) in enumerate(data):
        expected = scale * (1 + growth) ** i
        actual = block_size ** bs_fac * avg_fee ** fee_fac
        # if i >= len(data) - 6:
        #     err += ((expected / actual - 1) ** 2) * 2
        err += (expected / actual - 1) ** 2
    return err

best = (0, 0, 0, 9999999999999999999999999.0)

for scale in [1 * 1.05 ** x for x in range(300)]:
    for elasticity in [x*0.025 for x in range(120)]:
        for growth in [x*0.001 for x in range(120)]:
            err = get_error(scale, elasticity, growth)
            if err <= REPORT_THRESHOLD:
                print('%d %.3f %.3f: %.3f' % (scale, elasticity, growth, err))
            if err < best[-1]:
                best = scale, elasticity, growth, err

print('Best params: %d %.3f %.3f (err %.3f)' % best)

scale, elasticity, growth, err = best
bs_fac, fee_fac = 1 / (1 + elasticity), elasticity / (1 + elasticity)

for i, (block_size, avg_fee) in enumerate(data):
    expected = scale * (1 + growth) ** i
    actual = block_size ** bs_fac * avg_fee ** fee_fac
    print(i, actual, expected)
