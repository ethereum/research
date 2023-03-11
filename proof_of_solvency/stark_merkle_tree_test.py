import random
import time
from stark_merkle_tree import modulus, get_sum_trace, mk_por_proof, verify_por_proof, mk_inclusion_proof, verify_inclusion_proof, extension_factor, uts

USER_NUM = 2**8

def create_user_data():
    id = []
    balances = []
    for i in range(uts*USER_NUM):
        id.append(random.randrange(modulus) if i%uts==(uts-2) else 0)
        balances.append((random.randrange(2**(uts-2))//USER_NUM)*USER_NUM if i%uts==(uts-2) else 0)
    return id,balances

def test_stark_merkle_tree():
    id, balances = create_user_data()
    _, _, sum_amount = get_sum_trace(balances)
    proof, mtree = mk_por_proof(id, balances)
    assert verify_por_proof(USER_NUM*uts, sum_amount, balances[uts-2], proof)

    user_index = random.randrange(USER_NUM)
    time0 = time.time()
    inclusion_proof = mk_inclusion_proof(mtree, user_index)
    time1 = time.time()
    print("mk inclusion proof in %.4f sec" % (time1 - time0))
    assert verify_inclusion_proof(mtree[1], user_index, inclusion_proof)
    print("verify inclusion proof in %.4f sec" % (time.time() - time1))

def test_non_negative_constraint():
    id, balances = create_user_data()
    user_index = random.randrange(USER_NUM)

    balances[uts*user_index + (uts-2)] = -balances[uts*user_index + (uts-2)]
    _, _, sum_amount = get_sum_trace(balances)
    proof, _ = mk_por_proof(id, balances)
    try:
        verify_por_proof(USER_NUM*uts, sum_amount, balances[(uts-2)], proof)
    except AssertionError:
        print("Invalid proof")
