import random, heapq

def find_path(coins, frm, to, amount, online):
    neighbor_map = {}
    for i in range(amount, len(coins) - amount + 1):
        if coins[i-1] != coins[i]:
            if coins[i-1] in online and coins[i] in online:
                if coins[i-amount:i] == [coins[i-1]] * amount:
                    neighbor_map[coins[i-1]] = list(set(neighbor_map.get(coins[i-1], []) + [coins[i]]))
                if coins[i:i+amount] == [coins[i]] * amount:
                    neighbor_map[coins[i]] = list(set(neighbor_map.get(coins[i], []) + [coins[i-1]]))
    parents = {frm: None}
    q = [(0, frm)]
    while len(q) > 0:
        dist, sender = heapq.heappop(q)
        neighbors = neighbor_map.get(sender, [])
        for n in neighbors:
            if n not in parents:
                heapq.heappush(q, (dist+1, n))
                parents[n] = sender
                if n == to:
                    o = [n]
                    while o[0] != frm:
                        o.insert(0, parents[o[0]])
                    return o
    return False
            

def mk_fragmented_shuffle(coins, owners, shuffs):
    L = [(i * owners)//coins for i in range(coins)]
    for i in range(shuffs):
        x1 = random.randrange(n)
        x2 = random.randrange(n)
        value = int(min(n - x1, n - x2, abs(x2 - x1)) ** random.random())
        L[x1:x1+value], L[x2:x2+value] = L[x2:x2+value], L[x1:x1+value]
    return L
        
def mk_shuffle(n):
    L = list(range(1, n+1))
    random.shuffle(L)
    return ','.join([str(x) for x in L])

def fragments(vals):
    tot = 1
    for i in range(1, len(vals)):
        if vals[i] != vals[i-1]:
            tot += 1
    return tot

def send_coins(coins, frm, to, amt):
    coins_to_send = amt
    for i in range(len(coins)):
        if coins[i] == frm:
            coins[i] = to
            coins_to_send -= 1
        if coins_to_send == 0:
            return True
    return False

def shunt_coins(coins, frm, to, amt):
    i = 1
    while i < len(coins):
        while i < len(coins) and not((coins[i-1] == frm and coins[i] == to) or (coins[i-1] == to and coins[i] == frm)):
            i += 1
        if coins[i-amt:i] == [frm] * amt:
            coins[i-amt:i] = [to] * amt
            return True
        if coins[i:i+amt] == [frm] * amt:
            coins[i:i+amt] = [to] * amt
            return True
        i += 1
    return False

userz = 100
coinz = 50000
part_online = 0.2
c = [(i*userz)//coinz for i in range(coinz)]
for i in range(25000):
    if i%10 == 0:
        print(fragments(c))
    frm = random.randrange(userz)
    to = random.randrange(userz)
    # amount = int(c.count(frm) ** random.random())
    amount = random.randrange(1, 11)
    if frm == to or amount == 0:
        continue
    path = find_path(c, frm, to, amount+1, [i for i in range(userz) if random.random() < part_online or i in (frm, to)])
    #pre_balance = (c.count(frm), c.count(to))
    if path:
        print(path)
        assert path[0] == frm
        assert path[-1] == to
        for i in range(1, len(path)):
            assert shunt_coins(c, path[i-1], path[i], amount)
    else:
        print('no path')
        assert amount <= c.count(frm)
        assert send_coins(c, frm, to, amount)
    #post_balance = (c.count(frm), c.count(to))
    #assert pre_balance[0] - post_balance[0] == amount
    #assert post_balance[1] - pre_balance[1] == amount
