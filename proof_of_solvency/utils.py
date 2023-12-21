import sys
sys.path.append("..")
sys.path.append("../mimc_stark")
from mimc_stark.merkle_tree import blake
import pickle
import os

# Get the set of powers of R, until but not including when the powers
# loop back to 1
def get_power_cycle(r, modulus):
    o = [1, r]
    while o[-1] != 1:
        o.append((o[-1] * r) % modulus)
    return o[:-1]

# Extract pseudorandom indices from entropy
def get_pseudorandom_indices(seed, modulus, count, exclude_multiples_of=0):
    assert modulus < 2**24
    data = seed
    while len(data) < 4 * count:
        data += blake(data[-32:])
    if exclude_multiples_of == 0:
        return [int.from_bytes(data[i: i+4], 'big') % modulus for i in range(0, count * 4, 4)] 
    else:
        real_modulus = modulus * (exclude_multiples_of - 1) // exclude_multiples_of
        o = [int.from_bytes(data[i: i+4], 'big') % real_modulus for i in range(0, count * 4, 4)]
        return [x+1+x//(exclude_multiples_of-1) for x in o]

def is_a_power_of_2(x):
    return True if x==1 else False if x%2 else is_a_power_of_2(x//2)

def extend_user_data(ids, coins, uts):
    user_num = len(ids)
    coins_num = len(coins)
    for i in range(coins_num):
        assert(len(coins[i]) == user_num)

    extended_ids = [0] * uts
    extended_coins = []
    for i in range(user_num):
        extended_ids = extended_ids + ([0]*(uts-2) + [ids[i]] + [0])

    for i in range(coins_num):
        balances = [0] * uts
        for j in range(user_num):
            balances = balances + ([0]*(uts-2) + [coins[i][j]] + [0])
        extended_coins.append(balances)
    del balances
    return extended_ids,extended_coins

def get_sum_trace(coins, uts, modulus):
    user_num = len(coins[0])//uts

    trace = [0]*len(coins[0])
    sum_amounts = [0]*(len(coins)+1)

    # calculate sum_amounts for each coins and trace[uts * i + uts - 2] for each user
    for i in range(user_num):
        for j in range(len(coins)):
            sum_amounts[j] = (sum_amounts[j] + coins[j][i * uts + uts -2]) % modulus
            trace[uts * i + uts - 2] = (trace[uts * i + uts - 2] + coins[j][i * uts + uts - 2]) % modulus
    
    # calcualte total sum_amounts
    for i in range(len(coins)):
        sum_amounts[-1] += sum_amounts[i]

    # calculate trace[uts * i + uts - 1]
    for i in range(1, user_num):
        trace[uts * i + uts - 1] = (trace[uts * (i - 1) + uts - 1] + trace[uts * i + uts - 2]) % modulus

    # calculate trace[i] when i % uts != {uts-2, uts-1}
    for i in range(user_num):
        for j in range(uts-3):
            trace[uts*i+uts-3-j] = trace[uts*i+uts-2-j] // 3

    # calculate coins sum_amounts trace
    for i in range(len(coins)):
        for j in range(1, user_num):
            coins[i][uts * j + uts - 1] = (coins[i][uts * (j - 1) + uts - 1] + coins[i][uts * j + uts - 2]) % modulus

    return trace, sum_amounts, coins

def pad(ids, coins, max_user_num):
    padded_num = max_user_num
    user_num = len(ids)
    while padded_num > 2 * (user_num + 1):
        padded_num //= 2
    ids = ids + [0] * (padded_num - user_num - 1)
    for i in range(len(coins)):
        coins[i] = coins[i] + [0] * (padded_num - user_num - 1)
    return ids, coins

def get_entries(array):
    entries_len = len(array[0][0]) if type(array[0][0]) == list else len(array[0])
    entries = []
    for i in range(entries_len):
        x = b''
        for j in range(len(array)):
            if (type(array[j][0]) == list):
                for k in range(len(array[j])):
                    x = x + array[j][k][i].to_bytes(32, 'big')
            else:
                x = x + array[j][i].to_bytes(32, 'big')
        entries.append(x)
    del x,entries_len
    return entries

def calculate_l(seed, powers, array, modulus):
    l_len = len(array[0][0]) if type(array[0][0]) == list else len(array[0])
    k_len = 0
    l = []
    for i in range(len(array)):
        if type(array[i][0]) == int:
            k_len += 2 
        else: 
            k_len += 2 * len(array[i])
    k = [int.from_bytes(blake(seed + i.to_bytes(32,'big')), 'big') % modulus for i in range(k_len)]
    index = 0
    for i in range(l_len):
        x = 0
        for j in range(len(array)):
            if (type(array[j][0]) == list):
                for m in range(len(array[j])):
                    x = (x + k[index] * array[j][m][i] + k[index+1] * array[j][m][i] * powers[i]) % modulus
                    index = (index + 2) % k_len
            else:
                x = (x + k[index] * array[j][i] + k[index+1] * array[j][i] * powers[i]) % modulus
                index = (index + 2) % k_len
        l.append(x)
    del x,index,l_len,k_len
    return l

def verify_l(k, power, l, array, extra, modulus):
    index = 0
    for i in range(len(array)):
        if (type(array[i]) == list):
            for j in range(len(array[i])):
                l = (l - k[index] * array[i][j] - k[index+1] * array[i][j] * power ) % modulus
                index = index + 2

        else:
            l = (l - k[index] * array[i] - k[index+1] * array[i] * power) % modulus
            index = index + 2

    return (l - extra) % modulus == 0

def check_entry_hash(main_branch_leaves, mtree_entries_data, coins_num, modulus):
    user_random = [int.from_bytes(blake(data[len(data)-2*coins_num*32-4*32:]),'big') % modulus for data in mtree_entries_data]
    entries = [data[:len(data)-2*coins_num*32-4*32] + rd.to_bytes(32, 'big') for data, rd in zip(mtree_entries_data, user_random)]
    calculated_leaves = [blake(entry) for entry in entries]

    for i in range(len(main_branch_leaves)):
        assert main_branch_leaves[i] == calculated_leaves[i]

def save_data(data_path, sum_proof, mtree, mtree_entries_data, sum_amounts):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    with open(data_path + "/sum_proof.pickle", "wb") as ff:
        pickle.dump(sum_proof, ff)
    
    with open(data_path + "/mtree.pickle", "wb") as ff:
        pickle.dump(mtree, ff)

    with open(data_path + "/mtree_entries_data.pickle", "wb") as ff:
        pickle.dump(mtree_entries_data, ff)

    with open(data_path + "/sum_amounts.pickle", "wb") as ff:
        pickle.dump(sum_amounts, ff)

def check_in_field(x, modulus):
    for i in range(len(x)):
        if type(x[i]) == list: 
            for j in range(len(x[i])):
                x[i][j] = x[i][j] % modulus
        else:
            x[i] = x[i] % modulus

def check_sum_amounts(sum_amounts, modulus):
    cal_sum = 0
    for i in range(len(sum_amounts)-1):
        assert sum_amounts[i] >= 0 and sum_amounts[i] < modulus, "invalid sum amounts"
        cal_sum += sum_amounts[i]
    assert cal_sum == sum_amounts[-1]
