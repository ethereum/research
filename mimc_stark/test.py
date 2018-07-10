from fft import fft
from mimc_stark import mk_mimc_proof, modulus, mimc, verify_mimc_proof
from compression import compress_fri, compress_branches, bin_length
from merkle_tree import merkelize, mk_branch, verify_branch
from fri import prove_low_degree, verify_low_degree_proof

def test_merkletree():
    t = merkelize(range(128))
    b = mk_branch(t, 59)
    assert verify_branch(t[1], 59, b) == 59
    print('Merkle tree works')
    
def test_fri():
    # Pure FRI tests
    poly = list(range(512))
    root_of_unity = pow(7, (modulus-1)//1024, modulus)
    evaluations = fft(poly, modulus, root_of_unity)
    proof = prove_low_degree(poly, root_of_unity, evaluations, 512, modulus)
    print("Approx proof length: %d" % bin_length(compress_fri(proof)))
    assert verify_low_degree_proof(merkelize(evaluations)[1], root_of_unity, proof, 512, modulus)

def test_stark():
    INPUT = 3
    LOGSTEPS = 13
    LOGPRECISION = 16
    
    # Full STARK test
    proof = mk_mimc_proof(INPUT, LOGSTEPS, LOGPRECISION)
    p_root, d_root, k_root, b_root, l_root, branches, fri_proof = proof
    L1 = bin_length(compress_branches(branches))
    L2 = bin_length(compress_fri(fri_proof))
    print("Approx proof length: %d (branches), %d (FRI proof), %d (total)" % (L1, L2, L1 + L2))
    root_of_unity = pow(7, (modulus-1)//2**LOGPRECISION, modulus)
    subroot = pow(7, (modulus-1)//2**LOGSTEPS, modulus)
    skips = 2**(LOGPRECISION - LOGSTEPS)
    assert verify_mimc_proof(3, LOGSTEPS, LOGPRECISION, mimc(3, LOGSTEPS, LOGPRECISION), proof)
