import random, sys


def normal_distribution(mean, standev):
    def f():
        total = sum([random.choice([1, -1]) for i in range(529)])
        return int(total * (standev / 23.0) + mean)

    return f


def exponential_distribution(mean):
    def f():
        total = 0
        while 1:
            total += 1
            if not random.randrange(500):
                break
        return int(total * 0.002 * mean)

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
