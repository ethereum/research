from fft import fft
from mimc_stark import mk_mimc_proof, modulus, mimc, verify_mimc_proof
from merkle_tree import merkelize, mk_branch, verify_branch, bin_length
from fri import prove_low_degree, verify_low_degree_proof

def test_merkletree():
    t = merkelize([x.to_bytes(32, 'big') for x in range(128)])
    b = mk_branch(t, 59)
    assert verify_branch(t[1], 59, b, output_as_int=True) == 59
    print('Merkle tree works')

def fri_proof_bin_length(fri_proof):
    return sum([32 + bin_length(x[1]) + bin_length(x[2]) for x in fri_proof[:-1]]) + len(b''.join(fri_proof[-1]))
    
def test_fri():
    # Pure FRI tests
    poly = list(range(4096))
    root_of_unity = pow(7, (modulus-1)//16384, modulus)
    evaluations = fft(poly, modulus, root_of_unity)
    proof = prove_low_degree(evaluations, root_of_unity, 4096, modulus)
    print("Approx proof length: %d" % fri_proof_bin_length(proof))
    assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 4096, modulus)
    
    try:
        fakedata = [x if pow(3, i, 4096) > 400 else 39 for x, i in enumerate(evaluations)]
        proof2 = prove_low_degree(fakedata, root_of_unity, 4096, modulus)
        assert verify_low_degree_proof(merkelize(fakedata)[1], root_of_unity, proof, 4096, modulus)
        raise Exception("Fake data passed FRI")
    except:
        pass
    try:
        assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 2048, modulus)
        raise Exception("Fake data passed FRI")
    except:
        pass

def test_stark():
    INPUT = 3
    import sys
    LOGSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    # Full STARK test
    import random
    #constants = [random.randrange(modulus) for i in range(64)]
    constants = [(i**7) ^ 42 for i in range(64)]
    proof = mk_mimc_proof(INPUT, 2**LOGSTEPS, constants)
    m_root, l_root, main_branches, linear_comb_branches, fri_proof = proof
    L1 = bin_length(main_branches) + bin_length(linear_comb_branches)
    L2 = fri_proof_bin_length(fri_proof)
    print("Approx proof length: %d (branches), %d (FRI proof), %d (total)" % (L1, L2, L1 + L2))
    assert verify_mimc_proof(3, 2**LOGSTEPS, constants, mimc(3, 2**LOGSTEPS, constants), proof)

if __name__ == '__main__':
    test_stark()
