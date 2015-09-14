import random
BLKTIME = 600
LATENCY = 10
TEST_MAX = 1200
TEST_INTERVAL = 6

class Block():
    def __init__(self, parent, txstate):
        self.parent = parent
        self.score = 1 if parent is None else parent.score + 1
        if parent is None or parent.txstate is None:
            self.txstate = txstate
        else:
            self.txstate = parent.txstate


results = {}


for double_spend_delay in range(0, TEST_MAX, TEST_INTERVAL):
    results[double_spend_delay] = 0
    for _ in range(1000):
        a_head = None
        b_head = None
        recvqueue = {}
        for time_elapsed in range(5000):
            txstate = 1 if time_elapsed < double_spend_delay else 2
            # Miner A mines and sends (has 50% network share)
            if random.random() * BLKTIME < 0.5:
                a_head = Block(a_head, txstate)
                if time_elapsed + LATENCY not in recvqueue:
                    recvqueue[time_elapsed + LATENCY] = []
                recvqueue[time_elapsed + LATENCY].append(a_head)
            # Miner B mines and sends (has 50% network share)
            if random.random() * BLKTIME < 0.5:
                b_head = Block(b_head, txstate)
                if time_elapsed + LATENCY not in recvqueue:
                    recvqueue[time_elapsed + LATENCY] = []
                recvqueue[time_elapsed + LATENCY].append(b_head)
            # Receive blocks
            if time_elapsed in recvqueue:
                for b in recvqueue[time_elapsed]:
                    if not a_head or b.score > a_head.score or (b.score == a_head.score and random.random() < 0.5):
                        a_head = b
                    if not b_head or b.score > b_head.score or (b.score == b_head.score and random.random() < 0.5):
                        b_head = b
        # Check which transaction "made it"
        if a_head and a_head.txstate == 1:
            results[double_spend_delay] += 0.001
    print (double_spend_delay, results[double_spend_delay])

print(results)
