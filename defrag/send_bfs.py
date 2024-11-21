import random, heapq

# Assuming `online` is the set of users that is online, find a path to
# send `amount` coins from `frm` to `to` through `coins` where each
# step along the path is between users that have adjacent fragments.
# A transfer done in this way does not contribute to fragmentation.
def find_path(coins, frm, to, amount, online):
    # Determine who is whose neighbor
    neighbor_map = {}
    for i in range(amount, len(coins) - amount + 1):
        if coins[i-1] != coins[i]:
            if coins[i-1] in online and coins[i] in online:
                if coins[i-amount:i] == [coins[i-1]] * amount:
                    neighbor_map[coins[i-1]] = list(set(neighbor_map.get(coins[i-1], []) + [coins[i]]))
                if coins[i:i+amount] == [coins[i]] * amount:
                    neighbor_map[coins[i]] = list(set(neighbor_map.get(coins[i], []) + [coins[i-1]]))
    # Search for the path
    parents = {frm: None}
    q = [(0, frm)]
    while q:
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
            
# How many fragments are in this set of coins?
def fragments(vals):
    tot = 1
    for i in range(1, len(vals)):
        if vals[i] != vals[i-1]:
            tot += 1
    return tot

# Send `amt` coins from `frm` to `to`. Increases fragmentation by
# maximum 1
def send_coins(coins, frm, to, amt):
    coins_to_send = amt
    for i in range(len(coins)):
        if coins[i] == frm:
            coins[i] = to
            coins_to_send -= 1
        if coins_to_send == 0:
            return True
    return False

# Get the concrete range to transfer if we are transfering `amt`
# coins from `frm` to `to` (must be neighboring fragments)
def get_coin_shunt(coins, frm, to, amt):
    i = 1
    L = len(coins)
    while i < L:
        while i < L and coins[i] not in (frm, to):
            i += 1
        if not((coins[i-1] == frm and coins[i] == to) or (coins[i-1] == to and coins[i] == frm)):
            i += 1
            continue
        if coins[i-amt:i] == [frm] * amt and coins[i] == to:
            coins[i-amt:i] = [to] * amt
            return (i-amt, i, to)
        if coins[i:i+amt] == [frm] * amt and coins[i-1] == to:
            coins[i:i+amt] = [to] * amt
            return (i, i+amt, to)
        i += 1
    return False

# Find the largest slice controlled by `acct`
def maxslice(coins, acct):
    maxsz = 0
    sz = 0
    for i in range(len(coins)):
        if coins[i] == acct:
            sz += 1
            maxsz = max(sz, maxsz)
        else:
            sz = 0
    return maxsz

# Count the number of coins and the number of fragments
# held by each user
def count_coins_and_fragments(coins):
    user_count = max(coins) + 1
    coin_count = [0] * user_count
    frag_count = [0] * user_count
    for i in range(len(coins)):
        coin_count[coins[i]] += 1
        if i > 0 and coins[i] != coins[i-1]:
            frag_count[coins[i]] += 1
    return coin_count, frag_count

userz = 25
coinz = 50000
part_online = 0.1
initial_fragments_per_user = 100
ordering = list(range(userz)) * initial_fragments_per_user
random.shuffle(ordering)
c = [ordering[i * len(ordering) //coinz] for i in range(coinz)]
balances = count_coins_and_fragments(c)[0]
for i in range(250000):
    if i%100 == 0:
        print(i, fragments(c))
    # if i%2000 == 0:
    #     coin_count, frag_count = count_coins_and_fragments(c)
    #     print(sorted(zip(coin_count, frag_count)))

    # Randomly select sender, recipient and amount
    frm = random.randrange(userz)
    to = random.randrange(userz)
    if frm == to:
        continue
    pre_balance = balances[frm]
    amount = random.randrange(1, 1 + int(pre_balance ** random.random())) if pre_balance >= 2 else pre_balance
    full_amount = amount
    # print("Paying %d coins from %d to %d" % (amount, frm, to))
    # Randomly select the users that are online
    online = [i for i in range(userz) if random.random() < part_online or i in (frm, to)]
    while amount > 0:
        maxpay = maxslice(c, frm)
        pay_this_round = min(amount, maxpay)
        path = find_path(c, frm, to, pay_this_round, online)
        if path:
            #print("Found path for %d coins (%d hops)" % (pay_this_round, len(path)-1))
            assert path[0] == frm
            assert path[-1] == to
            shunts = []
            for i in range(1, len(path)):
                shunts.append(get_coin_shunt(c, path[i-1], path[i], pay_this_round))
                assert shunts[-1]
            for shunt in shunts:
                start, end, to = shunt
                c[start:end] = [to] * (end-start)
            amount -= pay_this_round
        else:
            # print('No path, paying remaining amount %d via fragmentation' % amount)
            # print('%d fragments' % fragments(c))
            assert send_coins(c, frm, to, amount)
            break
    balances[frm] -= full_amount
    balances[to] += full_amount
