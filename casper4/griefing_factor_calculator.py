# Computes griefing factors of various parameter sets for Casper the
# Friendly Finality Gadget

# Case 1: <1/3 non-commit (optimal if epsilon participate)
def gf1(x1, x2, x3, x4, x5):
    return x2 / x1

# Case 2: censor <1/3 committers (optimal if 1/3 get censored)
def gf2(x1, x2, x3, x4, x5):
    return 1.5 * (x1 + x2 / 3) / x2

# Generalized case 2
#k = 0.25
#def gf2(x1, x2, x3, x4, x5):
#    return (x1 * k + x2 * k**2) / (x2 * k * (1-k))

# Case 3: <1/3 non-prepare (optimal if epsilon participate)
def gf3(x1, x2, x3, x4, x5):
    return x4 / x3

# Case 4: censor <1/3 preparers (optimal if 1/3 get censored)
def gf4(x1, x2, x3, x4, x5):
    return 1.5 * (x3 + x4 / 3) / x4

# Case 5: finality preventing 1/3 non-commits
def gf5(x1, x2, x3, x4, x5):
    return 2 * (x5 + x2 / 3) / (x5 + x1 + x2 / 3)

# Case 6: censor commits
def gf6(x1, x2, x3, x4, x5):
    # Case 6a: 51% participate
    return max(1 + x2 / (x5 + x1 + x2 / 2),
    # Case 6b: 67% participate
               (x5 + x1 + x2 / 3) / (x5 + x2 / 3) / 2)

# Case 7: finality and commit-preventing 1/3 non-prepares
def gf7(x1, x2, x3, x4, x5):
    return 2 * (x5 + x4 / 3) / (x5 + x3 + x4 / 3)

gfs = (gf1, gf2, gf3, gf4, gf5, gf6, gf7)

# Get the maximum griefing factor of a set of parameters
def getmax(*args):
    return max([f(*args) for f in gfs])

# Get the maximum <50% griefing factor, and enforce a bound
# of MAX_CENSOR_GF on the griefing factor of >50% coalitions
def getmax2(*args):
    MAX_CENSOR_GF = 2
    if gf2(*args) > MAX_CENSOR_GF or gf4(*args) > MAX_CENSOR_GF or \
            gf6(*args) > MAX_CENSOR_GF:
        return 999999999999999999

    return max(gf1(*args), gf3(*args), gf5(*args), gf7(*args))

# Range to test for each parameter
my_range = [i/12. for i in range(1, 61)]

best_vals = (1, 0, 0, 0, 0)
best_score = 999999999999999999

# print([f(5, 6, 5, 6, 0) for f in gfs])

for x1 in my_range:
    for x2 in my_range:
        for x3 in my_range:
            for x4 in my_range:
                o = getmax2(x1, x2, x3, x4, 1)
                if o < best_score:
                    best_score = o
                    best_vals = (x1, x2, x3, x4, 1)
                if o <= 1:
                    print((x1, x2, x3, x4, 1), [f(x1, x2, x3, x4, 1) for f in gfs])
print('result', best_vals, best_score)
print([f(*best_vals) for f in gfs])
