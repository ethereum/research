import ec65536
import rlp

# 12.8 kilobyte test string
testdata = 'the cow jumped over the moon!!! ' * 400

prover = ec65536.Prover(testdata)
print("Created prover")

assert ec65536.verify_proof(prover.merkle_root, prover.prove(13), 13)

proofs = [prover.prove(i) for i in range(0, prover.length, 2)]
print("Created merkle proofs")

print("Starting to attempt fill")
response = ec65536.fill(prover.merkle_root, prover.length // 2, proofs, list(range(0, prover.length, 2)))
assert response is not False
assert b''.join(response)[:len(rlp.encode(testdata))] == rlp.encode(testdata)
print("Fill successful")
