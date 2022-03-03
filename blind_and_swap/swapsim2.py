import copy, sys, random

def merge_probs(*probs):
    o = {}
    for prob in probs: 
        for k,v in prob.items():
            o[k] = o.get(k, 0) + v / len(probs)
    return o

def run(width, rounds, swaps_per_round, extract_positions):
    array = [{i: 1} for i in range(width)]
    o = []
    for r in range(rounds):
        offsets = [(r*2+1)*i for i in range(swaps_per_round)]
        swap_indices = [(extract_positions[r-1] + offset) % width for offset in offsets]
        merged_probs = merge_probs(*[array[index] for index in swap_indices])
        for index in swap_indices:
            array[index] = copy.copy(merged_probs)
        extraction_index = extract_positions[r]
        o.append(array[extraction_index])
        array[extraction_index] = {width+r: 1}
    return o

def test(width, rounds, swaps_per_round):
    extract_positions = [random.randrange(width) for _ in range(rounds + swaps_per_round)]
    thresholds = []
    outputs = run(width, rounds, swaps_per_round, extract_positions)
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
    width, swaps_per_round = int(sys.argv[1]), int(sys.argv[2])
    rounds = width * 4
    thresholds = test(width, rounds, swaps_per_round)
    for i in range(0, rounds, 10):
        print("After {} rounds, need to DoS {} validators for 20% chance of killing proposer".format(i, thresholds[i]))
    second_half = thresholds[len(thresholds)//2:]
    print("Average of second half: {}".format(sum(second_half) / len(second_half)))
    print("Frequency of 1 in second half: {}".format(second_half.count(1) / len(second_half)))
