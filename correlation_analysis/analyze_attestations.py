import pickle
import os
import requests

ATTESTER_FILE = 'attesters.txt'
COMMITTEES_FILE = 'committees.txt'
CLUSTERS_FILE = 'clusters.csv'
FILES_URL = 'https://data.ethpandaops.io/efresearch/'

import bisect
import struct

# A frozenset-like object made out of a byte array. Slower but *way* less
# memory-intensive. Needed to run an analysis of this size.
class fast_set:
    def __init__(self, values):
        if not all(isinstance(x, int) and 0 <= x < 2**32 for x in values):
            raise ValueError("All values must be uint32.")
        self._bytes = b''.join(struct.pack('>I', v) for v in sorted(values))

    def __contains__(self, item):
        if not isinstance(item, int) or not 0 <= item < 2**32:
            return False
        packed_item = struct.pack('>I', item)
        index = bisect.bisect_left(self, packed_item)
        if index != len(self) and self[index] == packed_item:
            return True
        return False

    def __getitem__(self, index):
        return self._bytes[index*4:(index+1)*4]

    def __len__(self):
        return len(self._bytes) // 4

    def __iter__(self):
        for i in range(len(self)):
            yield struct.unpack('>I', self[i])[0]

    def intersection(self, other_set):
        result = set()
        for val in other_set:
            if val in self:
                result.add(val)
        return result

# If the output of `fun` is saved in `filename`, use that. Otherwise,
# execute `fun` and save at `filename`
def json_cached_execute(fun, filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except:
        o = fun()
        pickle.dump(o, open(filename, 'wb'))
        return o

# Output mapping {slot: set(validators who attested in that slot)}
def get_attesters_by_slot():
    attesters_by_slot = {}
    if ATTESTER_FILE not in os.listdir():
        print("Loading attesters file from internet")
        contents = requests.get(FILES_URL + ATTESTER_FILE).text
        open(ATTESTER_FILE, 'w').write(contents)
    print("Processing attesters file")
    for i, line in enumerate(open(ATTESTER_FILE).readlines()):
        if line:
            slot, attesters = line.split('\t')
            attesters_by_slot[int(slot)] = fast_set({
                int(x) for x in attesters.strip()[1:-1].split(',')
            })
        if i % 100 == 0:
            print("Processed {} lines".format(i))
    return attesters_by_slot

attesters_by_slot = json_cached_execute(get_attesters_by_slot, 'attesters_by_slot.pickle')
print('Processed attesters')

# Output mapping {slot: set(validators who should have attested in that slot)}
def get_committees_by_slot():
    committees_by_slot = {}
    if COMMITTEES_FILE not in os.listdir():
        print("Loading committees file from internet")
        contents = requests.get(FILES_URL + COMMITTEES_FILE).text
        open(COMMITTEES_FILE, 'w').write(contents)
    print("Processing committees file")
    for i, line in enumerate(open(COMMITTEES_FILE).readlines()):
        if line:
            slot, members = line.split('\t')
            committees_by_slot[int(slot)] = fast_set({
                int(x) for x in members.strip()[1:-1].split(',')
            })
        if i % 100 == 0:
            print("Processed {} lines".format(i))
    return committees_by_slot

committees_by_slot = json_cached_execute(get_committees_by_slot, 'committees_by_slot.pickle')
print("Processed committees")

def get_all_validators():
    all_validators = set()
    for committee in committees_by_slot.values():
        for value in committee:
            all_validators.add(value)
    return all_validators

all_validators = json_cached_execute(get_all_validators, 'all_validators.pickle')

print("Found all validators")

# Output mapping {attester: their cluster} and {cluster: set(attesters in cluster)}
# Gets a list of clusters (eg. 'coinbase', 'staked.us'...) from the provided file
def get_clusters():
    cluster_to_attesters = {}
    attester_to_cluster = {}
    
    if CLUSTERS_FILE not in os.listdir():
        print("Loading clusters file from internet")
        contents = requests.get(FILES_URL + CLUSTERS_FILE).text
        open(CLUSTERS_FILE, 'w').write(contents)
    print("Processing clusters file")
    for line in open(CLUSTERS_FILE).readlines():
        if line and line[:12] != 'validator_id':
            attester, cluster = line.strip().split(',')
            attester = int(attester)
            if attester not in all_validators:
                continue
            # If not in a cluster, the attester gets its own cluster
            cluster = cluster or attester
            if cluster not in cluster_to_attesters:
                cluster_to_attesters[cluster] = set()
            cluster_to_attesters[cluster].add(attester)
            attester_to_cluster[attester] = cluster

    for validator in all_validators:
        if validator not in attester_to_cluster:
            attester_to_cluster[validator] = validator
            cluster_to_attesters[validator] = {validator}
    return [cluster_to_attesters, attester_to_cluster]

cluster_to_attesters, attester_to_cluster = json_cached_execute(get_clusters, 'clusters.pickle')
print("Computed cluster data")

print('Processed cluster data. Found {} clusters'.format(len(cluster_to_attesters)))

min_slot = min(committees_by_slot.keys())
max_slot = max(committees_by_slot.keys())

# Track fumbles and misses for each slot

# Definition of a fumble: missed the current epoch but
# attested in the previous epoch
historical_fumbles = [0] * 32
# Definition of a miss: missed the current and previous epochs
historical_misses = [0] * 32

# Track totals
hits, fumbles, misses = 0, 0, 0

# "Co-fumbler" = pair of distinct validators in the same cluster that
# fumbles in the same epoch
co_fumblers = 0
expected_co_fumblers = 0
# "Co-miss" = pair of distinct validators in the same cluster that
# miss in the same epoch
co_missers = 0
expected_co_missers = 0

# For each of four penalty schemes, store the total penalties each
# validator receives according to that scheme
total_penalties_basic = {v: 0 for v in all_validators}
total_penalties_fumbles_only = {v: 0 for v in all_validators}
total_penalties_anticorr = {v: 0 for v in all_validators}
total_penalties_excess = {v: 0 for v in all_validators}

# Get the previous slot that `validator_id` was supposed to attest in.
# May return None if eg. the given slot is the validator's first active slot.
def get_prev_expected_slot(validator_id, slot):
    for s in range(slot - (slot % 32) - 1, slot - (slot % 32) - 33, -1):
        if validator_id in committees_by_slot[s]:
            return s

# Compute the statistics across our full range of slots
for slot in range(min_slot, max_slot):
    # Skip the first two epochs to make sure that there is a previous slot
    if slot not in committees_by_slot or slot - 63 not in committees_by_slot:
        continue
    this_slot_attesters = frozenset(x for x in attesters_by_slot[slot])
    fumblers = set()
    missers = set()
    committee_size = len(committees_by_slot[slot])
    # Sometimes the DB we're using screws up and returns a 2x too big committee
    # Skip those slots
    if committee_size > len(all_validators) / 24:
        print(f"Skipping slot {slot} (known miscalculated committee)")
        continue
    # Main loop: add fumbles and misses
    for validator in committees_by_slot[slot]:
        if validator in this_slot_attesters:
            hits += 1
        elif validator in attesters_by_slot.get(get_prev_expected_slot(validator, slot), {}):
            fumbles += 1
            fumblers.add(validator)
        else:
            misses += 1
            missers.add(validator)
    relative_fumbler_rate = len(fumblers) / ((sum(historical_fumbles[-32:]) + 1) / 32)
    relative_misser_rate = len(missers) / ((sum(historical_misses[-32:]) + 1) / 32)
    for fumbler in fumblers:
        total_penalties_basic[fumbler] += 1
        total_penalties_fumbles_only[fumbler] += 1
        total_penalties_anticorr[fumbler] += (len(fumblers) / (committee_size)) ** 0.2
        total_penalties_excess[fumbler] += max(0, min(relative_fumbler_rate - 1, 5))
        cluster_mates = cluster_to_attesters[attester_to_cluster[fumbler]]
        co_fumblers += len(fumblers.intersection(cluster_mates)) - 1
        # Expected co-fumblers = co-fumblers (pairs of fumblers in the same cluster)
        # if fumbling was truly uncorrelated
        expected_co_fumblers += (
            (len(cluster_mates) - 1) *
            (len(fumblers) - 1) / (len(all_validators) - 1)
        )
    for misser in missers:
        total_penalties_basic[fumbler] += 1
        total_penalties_anticorr[fumbler] += (len(fumblers) / (committee_size)) ** 0.2
        total_penalties_excess[fumbler] += max(0, min(relative_fumbler_rate - 1, 5))
        cluster_mates = cluster_to_attesters[attester_to_cluster[misser]]
        co_missers += len(missers.intersection(cluster_mates)) - 1
        expected_co_missers += (
            (len(cluster_mates) - 1) *
            (len(missers) - 1) / (len(all_validators) - 1)
        )
    historical_fumbles.append(len(fumblers))
    historical_misses.append(len(missers))
    print("Up to slot {}, {} fumbles and {} misses in this slot".format(
        slot, len(fumblers), len(missers)
    ))
    print("Cumulative stats: {} co-fumblers ({:.2f} expected), {} co-missers, ({:.2f} expected)".format(
        co_fumblers, expected_co_fumblers, co_missers, expected_co_missers
    ))

# Take a random selection of clusters at different sizes, chance of being
# selected is proportional to size, up to 100% for >= 10000
chosen_clusters = {
    k:v for k,v in cluster_to_attesters.items() if 
    (len(v) // 100 > len(v) % 100)
}
extra_clusters = {}
extra_clusters['Big clusters'] = set.union(
    *[v for v in cluster_to_attesters.values() if len(v) >= 10000]
)
extra_clusters['Medium clusters'] = set.union(
    *[v for v in cluster_to_attesters.values() if 100 <= len(v) < 10000]
)
extra_clusters['Small clusters'] = set.union(
    *[v for v in cluster_to_attesters.values() if 2 <= len(v) < 100]
)
chosen_clusters['Combined labeled'] = set.union(*chosen_clusters.values())
extra_clusters['All'] = all_validators

print("\nShowing simulated penalties for different clusters, using different penalty rules, normalized:\n")
shown_clusters = (
    sorted(extra_clusters.items(), key=lambda x: -len(x[1])) +
    sorted(chosen_clusters.items(), key=lambda x: -len(x[1]))
)
def average_for_set(source, validators):
    return sum(source[m] for m in validators) / len(validators)

global_avg_penalties_basic = average_for_set(total_penalties_basic, all_validators)
global_avg_penalties_fumbles_only = average_for_set(total_penalties_fumbles_only, all_validators)
global_avg_penalties_anticorr = average_for_set(total_penalties_anticorr, all_validators)
global_avg_penalties_excess = average_for_set(total_penalties_excess, all_validators)
for name, members in shown_clusters:
    print("{}: basic {:.3f}, fumbles only {:.3f}, anti-correlation {:.3f} excess {:.3f}".format(
        name,
        average_for_set(total_penalties_basic, members) / global_avg_penalties_basic,
        average_for_set(total_penalties_fumbles_only, members) / global_avg_penalties_fumbles_only,
        average_for_set(total_penalties_anticorr, members) / global_avg_penalties_anticorr,
        average_for_set(total_penalties_excess, members) / global_avg_penalties_excess,
    ))

print("\nTotal stats: {} hits ({:.2f}%), {} fumbles ({:.2f}%), {} misses ({:.2f}%)".format(
    hits, hits * 100 / (hits + fumbles + misses),
    fumbles, fumbles * 100 / (hits + fumbles + misses),
    misses, misses * 100 / (hits + fumbles + misses),
))
