import random, math

def mk_initial_balances(accts, coins):
    o = []
    for i in range(accts):
        o.extend([i] * random.randrange((coins - len(o)) * 2 // (accts - i)))
    o.extend([accts-1] * (coins - len(o)))
    return o

def fragments(coins):
    o = 0
    for i in range(1, len(coins)):
        if coins[i] != coins[i-1]:
            o += 1
    return o

def xfer(coins, frm, to, value):
    coins = coins[::]
    pos = 0
    while pos < len(coins) and value > 0:
        if coins[pos] == frm:
            coins[pos] = to
            value -= 1
        pos += 1
    return coins

def unscramble(coins, c1, c2):
    coins = coins[::]
    k1 = coins.count(c1)
    pos = 0
    while pos < len(coins):
        if coins[pos] in (c1, c2):
            coins[pos] = c1 if k1 > 0 else c2
            if coins[pos] == c1:
                k1 -= 1
        pos += 1
    return coins

def multi_unscramble(coins, addrs):
    coins = coins[::]
    ks = [coins.count(c) for c in addrs]
    pos = 0
    at = 0
    while pos < len(coins):
        if coins[pos] in addrs:
            coins[pos] = addrs[at]
            ks[at] -= 1
            if ks[at] == 0:
                at += 1
        pos += 1
    return coins

def unscramble_swap_strategy(coins, rounds):
    for i in range(rounds):
        c1, c2 = sorted([random.randrange(max(coins)+1) for _ in range(2)])
        coins = unscramble(coins, c1, c2)
    return coins

def run_with_unscrambling(coins, rounds):
    M = max(coins) + 1
    for i in range(rounds):
        c1, c2 = [random.randrange(M) for _ in range(2)]
        value = int(coins.count(c1) ** random.random())
        coins = xfer(coins, c1, c2, value)
        coins = unscramble(coins, min(c1, c2), max(c1, c2))
    return coins

def run_with_unscramble_online(coins, rounds):
    M = max(coins) + 1
    for i in range(rounds):
        c1, c2 = [random.randrange(M) for _ in range(2)]
        value = int(coins.count(c1) ** random.random())
        coins = xfer(coins, c1, c2, value)
        if random.random() < 1:
            cx = sorted([random.randrange(M) for _ in range(5)])
            coins = multi_unscramble(coins, cx)
    return coins

c = mk_initial_balances(200, 10000)
# random.shuffle(c)
# c = unscramble_swap_strategy(c, 20000)
c = run_with_unscramble_online(c, 10000)
print(fragments(c))
