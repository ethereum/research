from merkle_tree import merkelize, mk_multi_branch, verify_multi_branch

def test_multi_merkle_tree():
    tree = merkelize(list(range(16)))
    for i in range(65536):
        indices = [j for j in range(16) if (i>>j)%2 == 1]
        branch = mk_multi_branch(tree, indices)
        assert verify_multi_branch(tree[1], indices, branch) == [tree[16+j] for j in indices]
        if i%1024 == 1023:
            print("%d of 65536 16-element proofs checked" % (i+1))
    print("Multi Merkle tree test passed")

if __name__ == '__main__':
    test_multi_merkle_tree()
