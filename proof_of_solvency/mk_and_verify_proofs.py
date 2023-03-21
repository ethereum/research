import sys
sys.path.append("..")
sys.path.append("../mimc_stark")

from por_stark import mk_por_proof, verify_por_proof
from mimc_stark.permuted_tree import mk_branch, verify_branch
from mimc_stark.poly_utils import PrimeField
from constants import *
from utils import check_in_field, check_entry_hash

import random
import time
import json
import pickle

f = PrimeField(MODULUS)

def init_user_data():
    for batch in range(UTS16_BATCHES):
        data = []
        coins_len = len(COINS)
        for i in range(USER_NUM_INIT):
            items = {"id" : random.randrange(1000000000)}
            for coin in COINS:
                items[coin] = random.randrange(3**(UTS16 -2))//coins_len
            data.append(items)
        with open(USER_DATA_PATH + "batch" + str(batch) + ".json", "w") as f:
            json.dump(data, f)

    for batch in range(UTS16_BATCHES, UTS16_BATCHES + UTS32_BATCHES):
        data = []
        coins_len = len(COINS)
        for i in range(USER_NUM_INIT):
            items = {"id" : random.randrange(1000000000)}
            for coin in COINS:
                items[coin] = random.randrange(3**(UTS32 -2))//coins_len
            data.append(items)
        with open(USER_DATA_PATH + "batch" + str(batch) + ".json", "w") as f:
            json.dump(data, f)
    return

def read_user_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    ids = []
    balances = [[]]*(len(COINS))
    for item in data:
        ids.append(item["id"])
        j = 0
        for coin_name in COINS:
            if coin_name in item.keys():
                balances[j] = balances[j] + [item[coin_name]]
            else:
                balances[j] = balances[j] + [0]
            j += 1
    return ids, balances

def mk_batches_proof(start_batch = 0):
    if start_batch < UTS16_BATCHES:
        for i in range(start_batch, UTS16_BATCHES):
            ids, coins = read_user_data(USER_DATA_PATH + "batch" + str(i) + ".json")
            assert len(ids) < MAX_USER_NUM_FOR_UTS16, "too much users in one batch"
            data_path = BASIC_BATCH_PATH + "batch" + str(i)
            mk_por_proof(ids, coins, UTS16, data_path)

    for i in range(max(start_batch,UTS16_BATCHES), UTS16_BATCHES + UTS32_BATCHES):
        ids, coins = read_user_data(USER_DATA_PATH + "batch" + str(i) + ".json")
        assert len(ids) < MAX_USER_NUM_FOR_UTS32, "too much users in one batch"
        data_path = BASIC_BATCH_PATH + "batch" + str(i)
        mk_por_proof(ids, coins, UTS32, data_path)
    return

def verify_batches_proof():
    for i in range(UTS16_BATCHES + UTS32_BATCHES):
        with open(BASIC_BATCH_PATH + "batch" + str(i) + "/sum_proof.pickle", "rb") as ff:
            sum_proof = pickle.load(ff)

        with open(BASIC_BATCH_PATH + "batch" + str(i) + "/sum_amounts.pickle", "rb") as ff:
            sum_amounts = pickle.load(ff)

        assert verify_por_proof(sum_amounts, sum_proof), "invalid batch proof %d" % i

    return

def mk_trunk_proof():
    ids = []
    coins = [[]]*(len(COINS))
    for i in range(UTS16_BATCHES + UTS32_BATCHES):
        with open(BASIC_BATCH_PATH + "batch" + str(i) + "/mtree.pickle", "rb") as ff:
            ids.append(int.from_bytes(pickle.load(ff)[1], 'big') % MODULUS)
        
        with open(BASIC_BATCH_PATH + "batch" + str(i) + "/sum_amounts.pickle", "rb") as ff:
            sum_amounts = pickle.load(ff)
            j = 0
            for _ in COINS:
                coins[j] = coins[j] + [sum_amounts[j]]
                j += 1

    mk_por_proof(ids, coins, UTS_FOR_TRUNK, BASIC_TRUNK_PATH)
    return

def verify_trunk_proof():
    with open(BASIC_TRUNK_PATH + "/sum_proof.pickle", "rb") as ff:
        sum_proof = pickle.load(ff)

    with open(BASIC_TRUNK_PATH + "/sum_amounts.pickle", "rb") as ff:
        sum_amounts = pickle.load(ff)

    assert verify_por_proof(sum_amounts, sum_proof), "invalid trunk proof"

def mk_inclusion_proof(user_index, batch_index, batch_uts):
    with open(BASIC_BATCH_PATH + "batch" + str(batch_index) + "/mtree.pickle", "rb") as ff:
        batch_mtree = pickle.load(ff)
    with open(BASIC_BATCH_PATH + "batch" + str(batch_index) + "/mtree_entries_data.pickle", "rb") as ff:
        batch_mtree_entries_data = pickle.load(ff)   
    batch_inclusion_proof = [mk_branch(batch_mtree, (batch_uts * user_index + batch_uts-2) * EXTENSION_FACTOR), batch_mtree_entries_data[(batch_uts * user_index + batch_uts-2) * EXTENSION_FACTOR]]

    with open(BASIC_TRUNK_PATH + "/mtree.pickle", "rb") as ff:
        trunk_mtree = pickle.load(ff)
    with open(BASIC_TRUNK_PATH + "/mtree_entries_data.pickle", "rb") as ff:
        trunk_mtree_entries_data = pickle.load(ff)
    trunk_inclusion_proof = [mk_branch(trunk_mtree, (UTS_FOR_TRUNK * batch_index + UTS_FOR_TRUNK-2) * EXTENSION_FACTOR), trunk_mtree_entries_data[(UTS_FOR_TRUNK * batch_index + UTS_FOR_TRUNK-2) * EXTENSION_FACTOR]]

    return batch_inclusion_proof, trunk_inclusion_proof

def verify_inclusion_proof(root, index, uts, inclusion_proof, coins_num):
    leaf = verify_branch(root, (uts * index + uts-2) * EXTENSION_FACTOR, inclusion_proof[0])
    check_entry_hash([leaf], [inclusion_proof[1]], coins_num, MODULUS)
    return True

def test_inclusion_proof():
    user_index = random.randrange(USER_NUM_INIT)
    batch_index = random.randrange(UTS16_BATCHES)
    batch_inclusion_proof, trunk_inclusion_proof = mk_inclusion_proof(user_index, batch_index, UTS16)
    with open(BASIC_BATCH_PATH + "batch" + str(batch_index) + "/mtree.pickle", "rb") as ff:
        batch_root = pickle.load(ff)[1]
    with open(BASIC_TRUNK_PATH + "/mtree.pickle", "rb") as ff:
        trunk_root = pickle.load(ff)[1]
    assert verify_inclusion_proof(batch_root, user_index, UTS16, batch_inclusion_proof, len(COINS)), "invalid batch inclusion proof"
    assert verify_inclusion_proof(trunk_root, batch_index, UTS_FOR_TRUNK, trunk_inclusion_proof, len(COINS)),  "invalid trunk inclusion proof"

    user_index = random.randrange(USER_NUM_INIT)
    batch_index = random.randrange(UTS16_BATCHES, UTS16_BATCHES + UTS32_BATCHES)
    batch_inclusion_proof, trunk_inclusion_proof = mk_inclusion_proof(user_index, batch_index, UTS32)
    with open(BASIC_BATCH_PATH + "batch" + str(batch_index) + "/mtree.pickle", "rb") as ff:
        batch_root = pickle.load(ff)[1]
    assert verify_inclusion_proof(batch_root, user_index, UTS32, batch_inclusion_proof, len(COINS)), "invalid batch inclusion proof"
    assert verify_inclusion_proof(trunk_root, batch_index, UTS_FOR_TRUNK, trunk_inclusion_proof, len(COINS)), "invalid trunk inclusion proof" 

    print("Inclusion proof verified")
    return
