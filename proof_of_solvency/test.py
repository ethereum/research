from mk_and_verify_proofs import init_user_data, mk_batches_proof, verify_batches_proof, verify_batches_proof, mk_trunk_proof, \
                           verify_trunk_proof,  test_inclusion_proof

if __name__ == '__main__':
    init_user_data()
    mk_batches_proof()
    verify_batches_proof()
    mk_trunk_proof()
    verify_trunk_proof()
    test_inclusion_proof()