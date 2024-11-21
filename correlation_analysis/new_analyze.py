import os, sys, requests, pickle
import numpy as np

ATTESTER_FILE = 'attesters.txt'
IMMEDIATE_ATTESTER_FILE = 'immediate_attesters.txt'
COMMITTEES_FILE = 'committees.txt'
CLUSTERS_FILE = 'clusters.csv'
FILES_URL = 'https://data.ethpandaops.io/efresearch/'

def load_files():
    for file in (ATTESTER_FILE, IMMEDIATE_ATTESTER_FILE, COMMITTEES_FILE, CLUSTERS_FILE):
        if file not in os.listdir():
            print("Loading {}".format(file))
            contents = requests.get(FILES_URL + file).text
            open(file, 'w').write(contents)
        print("Loaded {}".format(file))

def get_participants_per_slot(participant_file):
    o = {}
    for i, line in enumerate(open(participant_file).readlines()):
        if line:
            slot, participants = line.split('\t')
            o[int(slot)] = np.array(participants.strip()[1:-1].split(','), dtype=np.uint32)
        if i%100 == 99:
            print(f"Processed {i+1} lines")
    print("Computed participants for {}".format(participant_file))
    return o

def get_participation_matrices():
    attesters_per_slot = get_participants_per_slot(ATTESTER_FILE)
    imm_attesters_per_slot = get_participants_per_slot(IMMEDIATE_ATTESTER_FILE)
    committees_per_slot = get_participants_per_slot(COMMITTEES_FILE)
    max_validator_id = max(max(v) for k,v in committees_per_slot.items())
    # If a slot has zero immediate attesters, it's a skipped slot, count it as
    # having full immediate attesters
    for slot in committees_per_slot:
        if slot not in imm_attesters_per_slot or len(imm_attesters_per_slot[slot]) == 0:
            imm_attesters_per_slot[slot] = committees_per_slot[slot]
    # Mapping all validators to a sequential index,
    # to make it easier to work with arrays
    # Only include validators who were active in both the first ten and
    # last ten epochs
    print("Computing set of all validators")
    min_slot = min(committees_per_slot)
    max_slot = max(committees_per_slot)
    all_validators_front = set()
    all_validators_back = set()
    for slot, committee in attesters_per_slot.items():
        if slot < min_slot + 320:
            for c in committee:
                all_validators_front.add(c)
        if slot >= max_slot - 320:
            for c in committee:
                all_validators_back.add(c)
    all_validators = all_validators_front.intersection(all_validators_back)
    validator_map = np.full(
        max_validator_id + 1, # length
        len(all_validators),     # fill with this value
        dtype=np.uint32
    )
    for i, v in enumerate(sorted(all_validators)):
        validator_map[v] = i
    # Sort away non-full epochs
    epochs = set((slot-31)//32 for slot in committees_per_slot)
    admissible_epochs = sorted(
        epoch for epoch in epochs if
        epoch-1 in epochs and epoch-2 in epochs
    )
    attesters = np.zeros((len(admissible_epochs), len(all_validators)+1), dtype=bool)
    imm_attesters = np.zeros((len(admissible_epochs), len(all_validators)+1), dtype=bool)
    slot_assignment = np.zeros((len(admissible_epochs), len(all_validators)+1), dtype=np.uint32)
    print("Processing epochs")
    for i, epoch in enumerate(admissible_epochs):
        for slot in range(epoch*32, epoch*32 + 32):    
            attesters[i][validator_map[attesters_per_slot[slot]]] = True
            imm_attesters[i][validator_map[imm_attesters_per_slot[slot]]] = True
            slot_assignment[i][validator_map[committees_per_slot[slot]]] = slot % 32
        if i % 10 == 9:
            print(f"Processed {i+1} epochs")
    print("Processed")
    return (attesters, imm_attesters, slot_assignment, validator_map)

# Output mapping {attester: their cluster} and {cluster: set(attesters in cluster)}
# Gets a list of clusters (eg. 'coinbase', 'staked.us'...) from the provided file
def get_clusters(validator_map):
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
            if attester >= len(validator_map):
                continue
            # If not in a cluster, the attester gets its own cluster
            cluster = cluster or attester
            if cluster not in cluster_to_attesters:
                cluster_to_attesters[cluster] = set()
            cluster_to_attesters[cluster].add(validator_map[attester])
            attester_to_cluster[validator_map[attester]] = cluster

    for validator in range(len(validator_map)):
        if validator not in attester_to_cluster:
            attester_to_cluster[validator] = validator
            cluster_to_attesters[validator] = {validator}
    return ({
        k: np.array(sorted(v)) for
        k,v in cluster_to_attesters.items()
    }, attester_to_cluster)

def get_fumbles(array):
    return array[:-1] * (1 - array[1:])

def get_misses(array):
    return (1 - array[:-1]) * (1 - array[1:])

def analyze_excess_correlation(array, cluster_to_attesters, slot_assignment):
    print("Analyzing excess correlations")
    assert len(array) + 1 == len(slot_assignment)

    def get_cofailures(attester_converter):
        total_processed = 0
        total_cluster_cofailures = 0
        for cluster, attesters in cluster_to_attesters.items():
            if len(attesters) > 10:
                vals = attester_converter(attesters)
                this_cluster_participation = array[:,vals]
                for slot in range(32):
                    this_cluster_this_slot = (slot_assignment[1:,vals] == slot)
                    failed_per_epoch = sum(np.transpose(this_cluster_participation * this_cluster_this_slot))
                    total_cluster_cofailures += sum(failed_per_epoch**2 - failed_per_epoch)
                total_processed += 1
                if total_processed % 25 == 24:
                    print(f"Processed {total_processed+1} clusters")
        return total_cluster_cofailures

    by_quality_order = np.argsort(sum(array))
    valcount = len(by_quality_order)
    position_in_quality_order = np.zeros(valcount, dtype=np.uint32)
    position_in_quality_order[by_quality_order] = np.arange(valcount)

    def get_fake_attesters(attesters):
        return by_quality_order[np.clip(position_in_quality_order[attesters] ^ 127, 0, valcount-1)]

    print("Co-failures: {}".format(get_cofailures(lambda x: x)))
    print("Fake co-failures: {}".format(get_cofailures(get_fake_attesters)))

def compute_excess_penalties(participation, slot_assignment):
    absences = np.ndarray((32, participation.shape[0]), dtype=np.uint32)
    for slot in range(32):
        expected_this_slot = (slot_assignment == slot)
        absences[slot] = sum((expected_this_slot * (1 - participation)).transpose()).astype(np.uint32)
    absences_in_prev_epoch = np.concatenate([np.array([0]), sum(absences)[:-1]])
    penalty_factor = np.clip(absences * 322 // absences_in_prev_epoch, 0, 4)
    print("Computed penalty factors")
    penalty = np.zeros(participation.shape[1], dtype=np.uint32)
    for slot in range(32):
        expected_this_slot = (slot_assignment == slot)
        absent_this_slot = expected_this_slot * (1 - participation)
        for i in range(penalty_factor.shape[1]):
            penalty[np.nonzero(absent_this_slot[i])] += penalty_factor[slot][i]
        print(f"Processed slot {slot} of 32")
    return penalty

def get_analysis_groups(cluster_to_attesters, validator_map):
    o = {
        'big':  np.concatenate([v for v in cluster_to_attesters.values() if len(v) >= 10000]),
        'medium': np.concatenate([v for v in cluster_to_attesters.values() if 300 < len(v) < 10000]),
        'small': np.concatenate([v for v in cluster_to_attesters.values() if 10 < len(v) <= 300]),
        'all': np.arange(max(validator_map)+1)
     }
    for cluster, values in cluster_to_attesters.items():
        if len(values) > 1000:
            o[cluster] = values
    return o

def run():
    if 'data.pickle' in os.listdir():
        data = pickle.load(open('data.pickle', 'rb'))
        print("Loaded data from pickle")
    else:
        data = get_participation_matrices()
        attesters, immediate_attesters, slot_assignment, validator_map = data
        print("Loaded participation matrices")
        data += get_clusters(validator_map)
        print("Loaded clusters")
        pickle.dump(data, open('data.pickle', 'wb'))
    (
        attesters, immediate_attesters, slot_assignment, validator_map,
        cluster_to_attesters, attester_to_cluster
    ) = data
    fumbles = get_fumbles(attesters)
    misses = get_misses(attesters)
    print("Fumbles (using max-deadline dataset):")
    analyze_excess_correlation(fumbles, cluster_to_attesters, slot_assignment)
    print("Misses (using max-deadline dataset):")
    analyze_excess_correlation(misses, cluster_to_attesters, slot_assignment)
    ssfumbles = get_fumbles(immediate_attesters)
    ssmisses = get_misses(immediate_attesters)
    print("Fumbles (using single-slot-deadline dataset):")
    analyze_excess_correlation(ssfumbles, cluster_to_attesters, slot_assignment)
    print("Misses (using single-slot-deadline dataset):")
    analyze_excess_correlation(ssmisses, cluster_to_attesters, slot_assignment)
    penalties = {
        'basic': sum(1 - attesters),
        'basic_ss': sum(1 - immediate_attesters),
        'excess': compute_excess_penalties(attesters, slot_assignment),
        'excess_ss': compute_excess_penalties(immediate_attesters, slot_assignment)
    }
    analysis_groups = get_analysis_groups(cluster_to_attesters, validator_map)
    first_col_width = max(len(x) for x in analysis_groups) + 2
    other_col_width = 15

    def fillto(text, k):
        return text + ' ' * (k - len(text))

    print(" " * first_col_width + ''.join(fillto(p, other_col_width) for p in penalties))
    for group, members in analysis_groups.items():
        o = fillto(group, first_col_width)
        for p in penalties:
            penalty = sum(penalties[p][members]) / len(members)
            o += fillto('{:.2f}'.format(penalty), other_col_width)
        print(o)

if __name__ == '__main__':
    run()
