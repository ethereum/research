# Get a Merkle proof
def _get_branch(db, node, keypath):
    if not keypath:
        return [db.get(node)]
    L, R, nodetype = parse_node(db.get(node))
    if nodetype == KV_TYPE:
        path = encode_bin_path(L)
        if keypath[:len(L)] == L:
            return [b'\x01'+path] + _get_branch(db, R, keypath[len(L):])
        else:
            return [b'\x01'+path, db.get(R)]
    elif nodetype == BRANCH_TYPE:
        if keypath[:1] == b0:
            return [b'\x02'+R] + _get_branch(db, L, keypath[1:])
        else:
            return [b'\x03'+L] + _get_branch(db, R, keypath[1:])

# Verify a Merkle proof
def _verify_branch(branch, root, keypath, value):
    nodes = [branch[-1]]
    _keypath = b''
    for data in branch[-2::-1]:
        marker, node = data[0], data[1:]
        # it's a keypath
        if marker == 1:
            node = decode_bin_path(node)
            _keypath = node + _keypath
            nodes.insert(0, encode_kv_node(node, sha3(nodes[0])))
        # it's a right-side branch
        elif marker == 2:
            _keypath = b0 + _keypath
            nodes.insert(0, encode_branch_node(sha3(nodes[0]), node))
        # it's a left-side branch
        elif marker == 3:
            _keypath = b1 + _keypath
            nodes.insert(0, encode_branch_node(node, sha3(nodes[0])))
        else:
            raise Exception("Foo")
        L, R, nodetype = parse_node(nodes[0])
    if value:
        assert _keypath == keypath
    assert sha3(nodes[0]) == root
    db = EphemDB()
    db.kv = {sha3(node): node for node in nodes}
    assert _get(db, root, keypath) == value
    return True
