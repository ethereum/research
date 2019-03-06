from merk import merkle_tree, mk_multi_proof, verify_multi_proof

def test_multi_merkle_tree():
    leaves = [i.to_bytes(32, 'big') for i in range(16)]
    tree = merkle_tree(leaves)
    for i in range(65536):
        indices = [j for j in range(16) if (i>>j)%2 == 1]
        proof = mk_multi_proof(tree, indices)
        assert verify_multi_proof(tree[1], indices, [leaves[i] for i in indices], 4, proof)
        if i%1024 == 1023:
            print("%d of 65536 16-element proofs checked" % (i+1))
            assert not verify_multi_proof(tree[2], indices, [leaves[i] for i in indices], 4, proof)
            assert not verify_multi_proof(tree[1], indices, [leaves[i][::-1] for i in indices], 4, proof)
    print("Multi Merkle tree test passed")

if __name__ == '__main__':
    test_multi_merkle_tree()
