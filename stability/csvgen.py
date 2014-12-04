import fit
import spread

tests = [
    [fit.simple_estimator, [1], [1]],
    [fit.simple_estimator, [1], [1.2]],
    [fit.minimax_estimator, [1], [1.2]],
    [fit.diff_estimator, [1, 0, 0.001], [1.2, 10, 1]],
    [fit.diff_estimator, [1, 0, 0.001, 0], [1.2, 10, 1, 5]],
    [fit.ndiff_estimator, [1, 0, 0, 0.001], [1.2, 10, 1, 1]],
    [fit.tx_diff_estimator, [1, 0, 0.001], [1.2, 10, 1]],
    [fit.tx_diff_estimator, [1, 0, 0.001, 0, 0], [1.2, 10, 1, 6, 2]],
    [fit.minimax_fee_estimator, [1, 3], [1.2, 60]],
]

vals = [fit.optimize(t, mi, ma, rate=0.4, rounds=12000, tries=10)
        for t, mi, ma in tests]

estimators = [t[0](*v) for t, v in zip(tests, vals)]

scores = [fit.evaluate_estimates(e) for e in estimators]

for t, v, e, s in zip(tests, vals, estimators, scores):
    print v, s

adjestimators = [[e/est[0] for e in est] for est in estimators]
adjprices = [[p/e*est[0] for e, p in 
             zip(est, fit.prices)] for est in estimators]
output = [['Price'] + fit.prices]
for i in range(len(tests)):
    output.append(['Estimator %d' % (i+1)] + adjestimators[i])
    output.append(['Adjusted price %d' % (i+1)] + adjprices[i])

spread.save('o.csv', zip(*output))
