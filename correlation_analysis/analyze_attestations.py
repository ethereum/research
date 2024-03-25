import pickle
import os
import requests

ATTESTER_FILE = 'attesters.txt'
IMMEDIATE_ATTESTER_FILE = 'immediate_attesters.txt'
COMMITTEES_FILE = 'committees.txt'
CLUSTERS_FILE = 'clusters.csv'
FILES_URL = 'https://data.ethpandaops.io/efresearch/'
MAX_VALIDATOR_ID = 1400000

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
def cached_load_from_web_db_file(filename):
    if filename not in os.listdir():
        print("Loading attesters file from internet")
        contents = requests.get(FILES_URL + filename).text
        open(filename, 'w').write(contents)
    if filename+'.pickle' not in os.listdir():
        print("Processing attesters file")
        obj = {}
        for i, line in enumerate(open(filename).readlines()):
            if line:
                slot, attesters = line.split('\t')
                obj[int(slot)] = fast_set({
                    int(x) for x in attesters.strip()[1:-1].split(',')
                })
            if i % 100 == 0:
                print("Processed {} lines".format(i))
        pickle.dump(obj, open(filename+'.pickle', 'wb'))
        return obj
    else:
        return pickle.load(open(filename+'.pickle', 'rb'))

immediate_attesters_by_slot = cached_load_from_web_db_file('immediate_attesters.txt')
print("Processed immediate attesters")
attesters_by_slot = cached_load_from_web_db_file('attesters.txt')
print('Processed attesters')
committees_by_slot = cached_load_from_web_db_file('committees.txt')
print('Processed committees')

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
# Like a fumble or miss but for the single-slot set
historical_ssfumbles = [0] * 32
historical_ssmisses = [0] * 32

# For (fumble, miss, ssfumble, ssmiss): instances of two distinct
# validators in the same cluster making this error within the same
# slot
co_totals = [0, 0, 0, 0]
expected_co_totals = [0, 0, 0, 0]

# For each of penalty schemes, store the total penalties each
# validator receives according to that scheme
schemes = ('basic', 'fumbles_only', 'ss_only', 'ss_fumblers_only', 'excess', 'ss_excess')
penalties = {
    scheme: {v: 0 for v in all_validators}
    for scheme in schemes
}

cluster_failures = {c: [0,0,0,0] for c in cluster_to_attesters}
processed_slots = 0

# Get the previous slot that `validator_id` was supposed to attest in.
# May return None if eg. the given slot is the validator's first active slot.
def get_prev_expected_slot(validator_id, slot):
    for s in range(slot - (slot % 32) - 1, slot - (slot % 32) - 33, -1):
        if validator_id in committees_by_slot[s]:
            return s

for slot in range(min_slot, max_slot):
    # Skip the first two epochs to make sure that there is a previous slot
    if slot not in committees_by_slot or slot - 63 not in committees_by_slot:
        continue
    # Skip slots where the next slot is empty
    if slot not in immediate_attesters_by_slot:
        continue

# Compute the statistics across our full range of slots
for slot in range(min_slot, max_slot):
    # Skip the first two epochs to make sure that there is a previous slot
    if slot not in committees_by_slot or slot - 63 not in committees_by_slot:
        continue
    # Skip slots where the next slot is empty
    if slot not in immediate_attesters_by_slot:
        continue
    this_slot_attesters = frozenset(x for x in attesters_by_slot[slot])
    this_slot_immediate_attesters = frozenset(x for x in immediate_attesters_by_slot[slot])
    # Definitions:
    # Fumble: missed in this epoch, attested in previous epoch
    # Miss: missed in this epoch and previous epoch
    # SSfumble (single-slot fumble): fumbles, but for the dataset of
    # attestations included within one slot
    # SSmiss: you get it
    fumblers = set()
    missers = set()
    ssfumblers = set()
    ssmissers = set()
    committee = committees_by_slot[slot]
    committee_size = len(committee)
    # Sometimes the DB we're using screws up and returns a 2x too big committee
    # Skip those slots
    if committee_size > len(all_validators) / 24:
        print(f"Skipping slot {slot} (known miscalculated committee)")
        del committees_by_slot[slot]
        continue
    # Main loop: add fumbles and misses
    for validator in committee:
        cluster = attester_to_cluster[validator]
        if validator not in this_slot_immediate_attesters:
            prev_slot = get_prev_expected_slot(validator, slot)
            if validator in immediate_attesters_by_slot.get(prev_slot, {}):
                ssfumblers.add(validator)
                cluster_failures[cluster][2] += 1
            else:
                ssmissers.add(validator)
                cluster_failures[cluster][3] += 1
            if validator not in this_slot_attesters:
                if validator in attesters_by_slot.get(prev_slot, {}):
                    fumblers.add(validator)
                    cluster_failures[cluster][0] += 1
                else:
                    missers.add(validator)
                    cluster_failures[cluster][1] += 1
    # Track co-fumbler, co-misser, etc stats
    for i, _set in enumerate((fumblers, missers, ssfumblers, ssmissers)):
        for member in _set:
            cluster_mates = cluster_to_attesters[attester_to_cluster[member]]
            co_totals[i] += len(_set.intersection(cluster_mates)) - 1
            expected_co_totals[i] += (
                (len(cluster_mates) - 1) *
                (len(_set) - 1) / (len(all_validators) - 1)
            )
    relative_failure_rate = (
        (len(fumblers) + len(missers)) /
        ((sum(historical_fumbles[-32:] + historical_misses[-32:]) + 1) / 32)
    )
    relative_ssfailure_rate = (
        (len(ssfumblers) + len(ssmissers)) /
        ((sum(historical_ssfumbles[-32:] + historical_ssmisses[-32:]) + 1) / 32)
    )
    # Now, apply the penalty schemes
    # schemes = ('basic', 'fumbles_only', 'ss_only', 'ss_fumbles_only', 'excess', 'ss_excess')
    for fumbler in fumblers:
        penalties['basic'][fumbler] += 1
        penalties['fumbles_only'][fumbler] += 1
        penalties['excess'][fumbler] += max(0, min(relative_failure_rate - 1, 4))
    for misser in missers:
        penalties['basic'][misser] += 1
        penalties['excess'][misser] += max(0, min(relative_failure_rate - 1, 4))
    for ssfumbler in ssfumblers:
        penalties['ss_only'][ssfumbler] += 1
        penalties['ss_fumblers_only'][ssfumbler] += 1
        penalties['ss_excess'][ssfumbler] += max(0, min(relative_ssfailure_rate - 1, 4))
    for ssmisser in ssmissers:
        penalties['ss_only'][ssmisser] += 1
        penalties['ss_excess'][ssmisser] += max(0, min(relative_ssfailure_rate - 1, 4))
    historical_fumbles.append(len(fumblers))
    historical_misses.append(len(missers))
    historical_ssfumbles.append(len(ssfumblers))
    historical_ssmisses.append(len(ssmissers))
    print("\nUp to slot {} ({} in committee): {} fumbles, {} misses, {} ssfumbles, {} ssmisses".format(
        slot, len(committee), len(fumblers), len(missers), len(ssfumblers), len(ssmissers)
    ))
    print("Cumulative stats: {} co-fumblers ({:.2f} expected), {} co-missers, ({:.2f} expected)".format(
        co_totals[0], expected_co_totals[0], co_totals[1], expected_co_totals[1]
    ))
    print("Single slot: {} co-fumblers ({:.2f} expected), {} co-missers, ({:.2f} expected)".format(
        co_totals[2], expected_co_totals[2], co_totals[3], expected_co_totals[3]
    ))
    processed_slots += 1

# Compute per-cluster failure rates

expected_co_totals2 = [0, 0, 0, 0]
for cluster, attesters in cluster_to_attesters.items():
    for i in range(4):
        failure_rate = cluster_failures[cluster][i] / (processed_slots / 32) / len(attesters)
        expected_co_totals2[i] += ((len(attesters) / 32) * failure_rate) ** 2 * processed_slots
print("Expectations: {:.2f} co-fumblers, {:.2f} co-missers, {:.2f} co-ssfumblers, {:.2f} co-ssmissers".format(*expected_co_totals2))


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
def average_for_set(source, validators):
    return sum(source[m] for m in validators) / len(validators)

global_avg_penalties = {
    scheme: average_for_set(penalties[scheme], all_validators)
    for scheme in schemes
}

shown_clusters = (
    sorted(extra_clusters.items(), key=lambda x: -len(x[1])) +
    sorted(chosen_clusters.items(), key=lambda x: -len(x[1]))
)

first_column_width = max(len(x[0]) for x in shown_clusters) + 2

def fillto(text, k):
    return text + ' ' * (k - len(text))

first_line = fillto('', first_column_width)
for scheme in schemes:
    first_line += fillto(scheme, len(scheme) + 2)
print(first_line)

for name, members in shown_clusters:
    line = fillto(name, first_column_width)
    for scheme in schemes:
        normalized_value = average_for_set(penalties[scheme], members) / global_avg_penalties[scheme]
        line += fillto('{:.3f}'.format(normalized_value), len(scheme) + 2)
    print(line)

fumbles = sum(historical_fumbles)
misses = sum(historical_misses)
ssfumbles = sum(historical_ssfumbles)
ssmisses = sum(historical_ssmisses)
total = sum(len(x) for x in committees_by_slot.values())

print("\nTotal stats: {} fumbles ({:.2f}%), {} misses ({:.2f}%), {} ssfumbles ({:.2f}%), {} ssmisses ({:.2f}%)".format(
    fumbles, fumbles * 100 / total,
    misses, misses * 100 / total,
    ssfumbles, ssfumbles * 100 / total,
    ssmisses, ssmisses * 100 / total,
))
