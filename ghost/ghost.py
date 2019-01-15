import os, random, time, struct
from binascii import hexlify

LATENCY_FACTOR = 0.5
NODE_COUNT = 131072
BLOCK_ONCE_EVERY = 1024
SIM_LENGTH = 131072

balances = [1] * NODE_COUNT
latest_message = [b'\x00' * 32] * NODE_COUNT
blocks = {b'\x00' * 32: (0, None)}
children = {}
ancestors = [{b'\x00' * 32: b'\x00' * 32} for i in range(16)]
max_known_height = [0]

logz = [0, 0]
for i in range(2, 10000):
    logz.append(logz[i // 2] + 1)
height_to_bytes = [i.to_bytes(4, 'big') for i in range(10000)]

def get_height(block):
    return blocks[block][0]

cache = {}
def get_ancestor(block, at_height):
    h = blocks[block][0]
    if at_height >= h:
        if at_height > h:
            return None
        else:
            return block
    cachekey = block + height_to_bytes[at_height]
    if cachekey in cache:
        return cache[cachekey]
    # assert get_height(ancestors[logz[h - at_height - 1]][block]) >= at_height
    o = get_ancestor(ancestors[logz[h - at_height - 1]][block], at_height)
    # assert get_height(o) == at_height
    cache[cachekey] = o
    return o

def add_block(parent):
    new_block_hash = os.urandom(32)
    h = get_height(parent)
    blocks[new_block_hash] = (h+1, parent)
    if parent not in children:
        children[parent] = []
    children[parent].append(new_block_hash)
    for i in range(16):
        if h % 2**i == 0:
            ancestors[i][new_block_hash] = parent
        else:
            ancestors[i][new_block_hash] = ancestors[i][parent]
    max_known_height[0] = max(max_known_height[0], h+1)

def add_attestation(block, validator_index):
    latest_message[validator_index] = block

def get_clear_winner(latest_votes, h):
    at_height = {}
    total_vote_count = 0
    for k, v in latest_votes.items():
        anc = get_ancestor(k, h)
        at_height[anc] = at_height.get(anc, 0) + v
        if anc is not None:
            total_vote_count += v
    for k, v in at_height.items():
        if v >= total_vote_count // 2:
            return k
    return None

def choose_best_child(votes):
    bitmask = 0
    for bit in range(255, -1, -1):
        zero_votes = 0
        one_votes = 0
        single_candidate = None
        for candidate in votes.keys():
            votes_for_candidate = votes[candidate]
            candidate_as_int = int.from_bytes(candidate, 'big')
            if candidate_as_int >> (bit+1) != bitmask:
                continue
            if (candidate_as_int >> bit) % 2 == 0:
                zero_votes += votes_for_candidate
            else:
                one_votes += votes_for_candidate
            if single_candidate is None:
                single_candidate = candidate
            else:
                single_candidate = False
        # print(bit, bitmask, zero_votes, one_votes)
        bitmask = (bitmask * 2) + (1 if one_votes > zero_votes else 0)
        if single_candidate:
            return single_candidate
    assert bit >= 1

def get_power_of_2_below(x):
    return 2**logz[x]

def ghost():
    # Get latest votes as key-value map
    latest_votes = {}
    for i in range(len(balances)):
        latest_votes[latest_message[i]] = latest_votes.get(latest_message[i], 0) + balances[i]

    head = b'\x00' * 32
    height = 0
    while 1:
        # print('at', height, 'votes', sum(latest_votes.values()))
        c = children.get(head, [])
        if len(c) == 0:
            return head
        step = get_power_of_2_below(max_known_height[0] - height) // 2
        while step > 0:
            possible_clear_winner = get_clear_winner(latest_votes, height - (height % step) + step)
            if possible_clear_winner is not None:
                # print("Skipping from height %d to %d" % (get_height(head), get_height(possible_clear_winner)))
                head = possible_clear_winner
                break
            step //= 2
        if step > 0:
            pass
        elif len(c) == 1:
            # print("Only child fast path", latest_votes.get(head, 0))
            head = c[0]
        else:
            # print("Block %s at height %d with %d children!" %
            #       (hexlify(head[:4]), height, len(c)), [hexlify(x[:4]) for x in c])
            child_votes = {x: 0.01 for x in c}
            for k, v in latest_votes.items():
                child = get_ancestor(k, height + 1)
                if child is not None:
                    child_votes[child] = child_votes.get(child, 0) + v
            head = choose_best_child(child_votes)
        height = get_height(head)
        deletes = []
        for k, v in latest_votes.items():
            if get_ancestor(k, height) != head:
                deletes.append(k)
        for k in deletes:
            del latest_votes[k]

def get_perturbed_head(head):
    up_count = 0
    while get_height(head) > 0 and random.random() < LATENCY_FACTOR:
        head = blocks[head][1]
        up_count += 1
    for _ in range(random.randrange(up_count + 1)):
        if head in children:
            head = random.choice(children[head])
    return head

def simulate_chain():
    start_time = time.time()
    for i in range(0, SIM_LENGTH, BLOCK_ONCE_EVERY):
        head = ghost()
        for j in range(i, i + BLOCK_ONCE_EVERY):
            phead = get_perturbed_head(head)
            add_attestation(phead, i % NODE_COUNT)
        print("Adding new block on top of block %d %s. Time so far: %.3f" %
              (blocks[phead][0], hexlify(phead[:4]), time.time() - start_time))
        add_block(phead)
            # print([get_height(latest_message[i]) for i in range(NODE_COUNT)])

simulate_chain()
print(len(str(cache)))
print(len(str(ancestors)))
print(len(str(blocks)))
