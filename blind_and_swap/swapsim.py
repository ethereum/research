import random, sys

def run(width, rounds, swaps_per_round, extract_positions):
    array = list(range(width))
    o = []
    for r in range(rounds):
        # Swap
        offsets = [(r*2+1)*i for i in range(swaps_per_round)]
        swap_indices = [(extract_positions[r-1] + offset) % width for offset in offsets]
        rotation = random.randrange(len(swap_indices))
        _buffer = [array[index] for index in swap_indices]
        _buffer = _buffer[rotation:] + _buffer[:rotation]
        for index, new_value in zip(swap_indices, _buffer):
            array[index] = new_value
        # Extract
        extraction_index = extract_positions[r]
        o.append(array[extraction_index])
        array[extraction_index] = width + r
    return o

def test(width, rounds, swaps_per_round, runs):
    extract_positions = [random.randrange(width) for _ in range(rounds + swaps_per_round)]
    outputs = [{} for _ in range(rounds)]
    for r in range(runs):
        output = run(width, rounds, swaps_per_round, extract_positions)
        if r % 10 == 0:
            print("Round {}".format(r))
        for store, val in zip(outputs, output):
            store[val] = store.get(val, 0) + 1
    thresholds = []
    for store in outputs:
        top_freqs = sorted(store.values(), reverse=True)
        # print(top_freqs)
        await_count = sum(top_freqs) * 0.2
        for i in range(len(top_freqs)):
            await_count -= top_freqs[i]
            if await_count <= 0:
                thresholds.append(i+1)
                break
        if thresholds[-1] == 1:
            print(top_freqs[:20])
        # print(thresholds[-1])
    return thresholds

if __name__ == '__main__':
    width, swaps_per_round, runs = int(sys.argv[1]), int(sys.argv[2]), 1000
    rounds = width * 4
    thresholds = test(width, rounds, swaps_per_round, runs)
    for i in range(0, rounds, 10):
        print("After {} rounds, need to DoS {} validators for 20% chance of killing proposer".format(i, thresholds[i]))
    second_half = thresholds[len(thresholds)//2:]
    print("Average of second half: {}".format(sum(second_half) / len(second_half)))
    print("Frequency of 1 in second half: {}".format(second_half.count(1) / len(second_half)))
