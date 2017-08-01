from bin_utils import encode_bin_path, decode_bin_path, common_prefix_length, encode_bin, decode_bin
from ethereum.utils import sha3, encode_hex

class EphemDB():
    def __init__(self):
        self.kv = {}

    def get(self, k):
        return self.kv.get(k, None)

    def put(self, k, v):
        self.kv[k] = v

KV_TYPE = 0
BRANCH_TYPE = 1

b1 = bytes([1])
b0 = bytes([0])

def parse_node(node):
    if len(node) == 64:
        return node[:32], node[32:], BRANCH_TYPE
    else:
        return decode_bin_path(node[:-32]), node[-32:], KV_TYPE

def encode_kv_node(keypath, node):
    assert keypath
    assert len(node) == 32
    o = encode_bin_path(keypath) + node
    assert len(o) < 64
    return o

def encode_branch_node(left, right):
    assert len(left) == len(right) == 32
    return left + right

def hash_and_save(db, node):
    h = sha3(node)
    db.put(h, node)
    return h

def _get(db, node, keypath):
    if not keypath:
        return db.get(node)
    L, R, nodetype = parse_node(db.get(node))
    if nodetype == KV_TYPE:
        if keypath[:len(L)] == L:
            return _get(db, R, keypath[len(L):])
        else:
            return None
    elif nodetype == BRANCH_TYPE:
        if keypath[:1] == b0:
            return _get(db, L, keypath[1:])
        else:
            return _get(db, R, keypath[1:])

def _update(db, node, keypath, val):
    if not keypath:
        return hash_and_save(db, val)
    if not node:
        return hash_and_save(db, encode_kv_node(keypath, hash_and_save(db, val)))
    L, R, nodetype = parse_node(db.get(node))
    if nodetype == KV_TYPE:
        if keypath[:len(L)] == L:
            o = _update(db, R, keypath[len(L):], val)
            assert o is not None
            if len(L) == len(keypath):
                return hash_and_save(db, encode_kv_node(L, o)) if o else b''
            subL, subR, subnodetype = parse_node(db.get(o))
            if subnodetype == KV_TYPE:
                return hash_and_save(db, encode_kv_node(L + subL, subR))
            else:
                return hash_and_save(db, encode_kv_node(L, o)) if o else b''
        else:
            cf = common_prefix_length(L, keypath[:len(L)])
            if len(keypath) == cf + 1:
                valnode = val
            else:
                valnode = hash_and_save(db, encode_kv_node(keypath[cf+1:], hash_and_save(db, val)))
            if len(L) == cf + 1:
                oldnode = R
            else:
                oldnode = hash_and_save(db, encode_kv_node(L[cf+1:], R))
            if keypath[cf:cf+1] == b1:
                newsub = hash_and_save(db, encode_branch_node(oldnode, valnode))
            else:
                newsub = hash_and_save(db, encode_branch_node(valnode, oldnode))
            if cf:
                return hash_and_save(db, encode_kv_node(L[:cf], newsub))
            else:
                return newsub
    elif nodetype == BRANCH_TYPE:
        newL, newR = L, R
        if keypath[:1] == b0:
            newL = _update(db, L, keypath[1:], val)
        else:
            newR = _update(db, R, keypath[1:], val)
        if not newL or not newR:
            subL, subR, subnodetype = parse_node(db.get(newL or newR))
            first_bit = b1 if newR else b0
            if subnodetype == KV_TYPE:
                return hash_and_save(db, encode_kv_node(first_bit + subL, subR))
            elif subnodetype == BRANCH_TYPE:
                return hash_and_save(db, encode_kv_node(first_bit, newL or newR))
            raise Exception("cow")
        else:
            return hash_and_save(db, encode_branch_node(newL, newR))
    raise Exception("cow")

def print_and_check_invariants(db, node, prefix=b''):
    #print('pci', node, prefix)
    if len(prefix) == 160:
        return {prefix: db.get(node)}
    if node == b'' and prefix == b'':
        return {}
    L, R, nodetype = parse_node(db.get(node))
    #print('lrn', L, R, nodetype)
    if nodetype == KV_TYPE:
        assert 0 < len(L) <= 160 - len(prefix)
        if len(L) + len(prefix) < 160:
            subL, subR, subnodetype = parse_node(db.get(R))
            assert subnodetype != KV_TYPE
        return print_and_check_invariants(db, R, prefix + L)
    else:
        assert L and R
        o = {}
        o.update(print_and_check_invariants(db, L, prefix + b0))
        o.update(print_and_check_invariants(db, R, prefix + b1))
        return o

def print_nodes(db, node, prefix=b''):
    if len(prefix) == 160:
        print('value node', encode_hex(node[:4]), db.get(node))
        return
    if node == b'':
        print('empty node')
        return
    L, R, nodetype = parse_node(db.get(node))
    if nodetype == KV_TYPE:
        print(('kv node:', encode_hex(node[:4]), ''.join(['1' if x == 1 else '0' for x in L]), encode_hex(R[:4])))
        print_nodes(db, R, prefix + L)
    else:
        print(('branch node:', encode_hex(node[:4]), encode_hex(L[:4]), encode_hex(R[:4])))
        print_nodes(db, L, prefix + b0)
        print_nodes(db, R, prefix + b1)

def _get_branch(db, node, keypath):
    if not keypath:
        return [node]
    L, R, nodetype = parse_node(db.get(node))
    if nodetype == KV_TYPE:
        path = encode_bin_path(L)
        if keypath[:len(L)] == L:
            return [path] + _get_branch(db, R, keypath[len(L):])
        else:
            return [path]
    elif nodetype == BRANCH_TYPE:
        if keypath[:1] == b0:
            return [R] + _get_branch(db, L, keypath[1:])
        else:
            return [L] + _get_branch(db, R, keypath[1:])

class Trie():
    def __init__(self, db, root):
        self.db = db
        self.root = root
        assert isinstance(self.root, bytes)

    def get(self, key):
        assert len(key) == 20
        return _get(self.db, self.root, encode_bin(key))

    def get_branch(self, key):
        return _get_branch(self.db, self.root, encode_bin(key))

    def update(self, key, value):
        assert len(key) == 20
        self.root = _update(self.db, self.root, encode_bin(key), value)

    def to_dict(self, hexify=False):
        o = print_and_check_invariants(self.db, self.root)
        encoder = lambda x: encode_hex(x) if hexify else x
        return {encoder(decode_bin(k)): v for k, v in o.items()}

    def print_nodes(self):
        print_nodes(self.db, self.root)        
