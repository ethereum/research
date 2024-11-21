import time
import sys

"Simple wrapper to measure the timer of a function. Can be used either as a decorator, or as f = chrono(f)"
def chrono(f):
    def timed(*args, **kargs):
        t = -time.time()
        result = f(*args, **kargs)
        t += time.time()

        print(f'func {f.__name__} exec in {t:2.6f}s', file=sys.stderr)
        return result
    return timed

if __name__ == '__main__':

    @chrono
    def testFunc(n):
        from math import factorial
        return factorial(n)

    testFunc(10**5)
    
