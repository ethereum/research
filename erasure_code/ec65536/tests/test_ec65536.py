import ec65536
import rlp
import time

# 12.8 kilobyte test string
testdata = 'the cow jumped over the moon!!! ' * 400

t1 = time.time()
prover = ec65536.Prover(testdata)
t2 = time.time()
print("Created prover in %.2f sec" % (t2 - t1))

assert ec65536.verify_proof(prover.merkle_root, prover.prove(13), 13)
t3 = time.time()
print("Created and verified a proof in %.2f sec" % (t3 - t2))

proofs = [prover.prove(i) for i in range(0, prover.length, 2)]
print("Created merkle proofs")

t4 = time.time()
print("Starting to attempt fill")
response = ec65536.fill(prover.merkle_root, prover.length // 2, proofs, list(range(0, prover.length, 2)))
t5 = time.time()
print("Completed fill in %.2f sec" % (t5 - t4))
assert response is not False
assert b''.join(response)[:len(rlp.encode(testdata))] == rlp.encode(testdata)
print("Fill successful")
