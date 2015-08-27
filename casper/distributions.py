import random, sys


def normal_distribution(mean, standev):
    def f():
        return int(random.normalvariate(mean, standev))
        # total = sum([random.choice([2, 0, 0, -2]) for i in range(8)])
        # return int(total * (standev / 4.0) + mean)

    return f


def exponential_distribution(mean):
    def f():
        total = 0
        while 1:
            total += 1
            if not random.randrange(32):
                break
        return int(total * 0.03125 * mean)

    return f


def convolve(*args):
    def f():
        total = 0
        for arg in args:
            total += arg()
        return total

    return f


def transform(dist, xformer):
    def f():
        return xformer(dist())

    return f
