from ethereum.utils import sha3
import sys

STEPLENGTH = 100

dp = {}

def step(inp):
    return sha3('function_'+inp.encode('hex')+'()')[:4]

def run_round(inp):
    orig_inp = inp
    for i in range(STEPLENGTH):
        inp = step(inp)
        if inp in dp.keys():
            print 'Found!', i + 1, repr(inp)
            return(True, i + 1, inp)
    dp[inp] = orig_inp
    return(False, None, inp)

y = '\xff' * 4
orig_y = y
rounds = 0
while 1:
    print 'Running round', rounds
    rounds += 1
    x, t, y2 = run_round(y)
    if x:
        prev1, prev2 = y, dp[y2]
        assert prev1 != prev2
        # print '-----'
        for i in range(STEPLENGTH - t):
            # print repr(prev2)
            prev2 = step(prev2) 
        # print '-----'
        for i in range(t):
            # print repr(prev1), repr(prev2)
            next1 = step(prev1)
            next2 = step(prev2)
            if next1 == next2:
                print 'Found!'
                print 'function_'+prev1.encode('hex')+'()'
                print 'function_'+prev2.encode('hex')+'()'
                sys.exit()
            prev1, prev2 = next1, next2
        # print repr(prev1), repr(prev2)
        raise Exception("Something weird happened")
    else:
        y = y2
