from new_bintrie import EphemDB, new_tree, get, update, make_merkle_proof, verify_proof
import random
from ethereum.utils import sha3

db = EphemDB()
t = new_tree(db)
for i in range(500):
    t = update(db, t, sha3(str(i)), sha3(str(i**3)))
for i in range(500):
    assert get(db, t, sha3(str(i))) == sha3(str(i**3))
for i in range(501, 1000):
    assert get(db, t, sha3(str(i))) == b'\x00' * 32

for i in range(1000):
    key = sha3(str(i))
    value = sha3(str(i ** 3)) if i < 500 else b'\x00' * 32
    proof = make_merkle_proof(db, t, key)
    assert verify_proof(proof, t, key, value)
