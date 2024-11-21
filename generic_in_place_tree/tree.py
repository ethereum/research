import leveldb
import hashlib
import sys

ZERO = b'\x00'*32


def hash(x):
    return hashlib.sha256(x).digest()


def hash_to_display_form(h):
    return '0x'+h.hex()[:12] if h else None


def get_common_prefix(a, b):
    L = 0
    while L < len(a) and L < len(b) and a[L] == b[L]:
        L += 1
    return a[:L]


class LeafNode():
    
    def __init__(self, key, value):
        assert len(key) == 32
        self.key, self.value = key, value

    def serialize(self):
        return b'L' + self.key + self.value

    def __repr__(self):
        return "Leaf[Key={}, value={}, hash={}]".format(self.key.rstrip(b'\x00'), self.value.rstrip(b'\x00'), hash_to_display_form(hash_node(self)))


class BranchNode():

    def __init__(self, values={}):
        if isinstance(values, list):
            assert len(values) == 256
            self.values = values
        elif isinstance(values, dict):
            self.values = [ZERO] * 256
            for k, v in values.items():
                self.values[k] = v
        else:
            raise Exception("Invalid values for branch node")

    def serialize(self):
        if self.values.count(ZERO) == 256:
            return None
        else:
            return b'B' + b''.join(self.values)

    def __repr__(self):
        if self.values.count(ZERO) == 256:
            return "BranchNode[]"
        o = "BranchNode["
        for i in range(256):
            if self.values[i] != ZERO:
                o += '{}: {}, '.format(chr(i), hash_to_display_form(self.values[i]))
        o += 'hash=' + hash_to_display_form(hash_node(self)) + ']'
        return o


def deserialize(node):
    if node is None:
        return None
    elif node[0] == ord(b'L'):
        return LeafNode(node[1:33], node[33:])
    elif node[0] == ord(b'B'):
        assert len(node) == 1 + 32 * 256
        return BranchNode([node[32*i+1:32*i+33] for i in range(256)])
    else:
        raise Exception("Broken node in DB: {}".format(node))


def hash_node(node):
    if isinstance(node, LeafNode):
        return hash(node.serialize())
    elif isinstance(node, BranchNode):
        # replace with Kate commitment
        serialized = node.serialize()
        return None if serialized is None else hash(serialized)
    elif node is None:
        return None
    else:
        raise Exception("Bad node type")


def db_get(db, key):
    try:
        return db.Get(key)
    except:
        return None


def db_put(db, key, value):
    print('putting', key, deserialize(value))
    if value is not None:
        db.Put(key, value)
    else:
        db.Delete(key)


def propagate_deletions(db, batch, path):

    for i in reversed(range(len(path))):
        current_node = deserialize(db_get(db, path[:i]))
        print(path[:i], current_node)
        assert isinstance(current_node, BranchNode)
        if current_node.values.count(ZERO) == 255:
            print('one nonzero; continuing')
            db_put(batch, path[:i], None)
        elif current_node.values.count(ZERO) == 254:
            print('two nonzeroes; replacing with leaf')
            db_put(batch, path[:i], sister_leaf.serialize())
            propagate_along_path(db, batch, path[:i], hash_node(sister_leaf))
            return
        else:
            print('3+ nonzeroes; removing')
            current_node.values[path[i]] = ZERO
            db_put(batch, path[:i], current_node.serialize())
            propagate_along_path(db, batch, path[:i], hash_node(current_node))
            return


def propagate_along_path(db, batch, path, new_node_hash):
    for i in reversed(range(len(path))):
        current_node = deserialize(db_get(db, path[:i]))
        if current_node is None or isinstance(current_node, LeafNode):
            current_node = BranchNode()
        current_node.values[path[i]] = new_node_hash or ZERO
        db_put(batch, path[:i], current_node.serialize())
        new_node_hash = hash_node(current_node)
    db.Write(batch)
                
        

def add(db, key, value):
    print('## ADDING {} {} ##'.format(key.rstrip(b'\x00'), value.rstrip(b'\x00')))
    assert len(key) == 32
    batch = leveldb.WriteBatch()
    for i, byte in enumerate(key):
        path = key[:i]
        node_at_path = deserialize(db_get(db, path))
        if node_at_path is None:
            new_leaf = LeafNode(key, value)
            db_put(batch, path, new_leaf.serialize())
            propagate_along_path(db, batch, path, hash_node(new_leaf))
            return
        if isinstance(node_at_path, LeafNode):
            new_leaf = LeafNode(key, value)
            if node_at_path.key == key:
                db_put(batch, path, new_leaf.serialize())
                propagate_along_path(db, batch, path, hash_node(new_leaf))
            else:
                propagation_path = get_common_prefix(key, node_at_path.key)
                common_prefix_length = len(propagation_path)
                db_put(batch, key[:common_prefix_length+1], new_leaf.serialize())
                db_put(batch, node_at_path.key[:common_prefix_length+1], node_at_path.serialize())
                new_branch_node = BranchNode({
                    key[common_prefix_length]: hash_node(new_leaf),
                    node_at_path.key[common_prefix_length]: hash_node(node_at_path)
                })
                db_put(batch, key[:common_prefix_length], new_branch_node.serialize())
                propagate_along_path(db, batch, propagation_path, hash_node(new_branch_node))
            return
    raise Exception("How did we get here?")

def delete(db, key):
    print('## DELETING {} ##'.format(key.rstrip(b'\x00')))
    assert len(key) == 32
    batch = leveldb.WriteBatch()
    node_at_path = None
    for i, byte in enumerate(key):
        path = key[:i]
        penultimate_node = node_at_path
        node_at_path = deserialize(db_get(db, path))
        if node_at_path is None:
            return
        if isinstance(node_at_path, LeafNode):
            if node_at_path.key == key:   
                db_put(batch, path, None)
                penultimate_path = key[:i-1]
                # Trivial case: last key is being removed
                if penultimate_node is None:
                    db.Write(batch)
                    return
                # Invariant: the immediately prior branch node cannot have only one child
                assert isinstance(penultimate_node, BranchNode)
                assert penultimate_node.values.count(ZERO) <= 254
                # Easy case: > 2 children
                if penultimate_node.values.count(ZERO) < 254:
                    penultimate_node.values[key[i-1]] = ZERO
                    db_put(batch, penultimate_path, penultimate_node.serialize())
                    propagate_along_path(db, batch, penultimate_path, hash_node(penultimate_node))
                    return
                # Hard case: 2 children
                sister_leaf_index = min(
                    j for j in range(256) if penultimate_node.values[j] != ZERO and j != key[i-1]
                )
                sister_leaf = deserialize(db_get(db, penultimate_path + bytes([sister_leaf_index])))
                db_put(batch, penultimate_path, None)
                for j in reversed(range(i-1)):
                    path = key[:j]
                    node_at_path = deserialize(db_get(db, path))
                    assert isinstance(node_at_path, BranchNode)
                    if node_at_path.values.count(ZERO) == 255:
                        db_put(batch, path, None)
                    else:
                        node_at_path.values[key[j]] = hash_node(sister_leaf)
                        db_put(batch, path + bytes([key[j]]), sister_leaf.serialize())
                        db_put(batch, path, node_at_path.serialize())
                        propagate_along_path(db, batch, path, hash_node(node_at_path))
                        return
                db_put(batch, b'', sister_leaf.serialize())
                db.Write(batch)
                return
            else:
                return
    raise Exception("How did we get here?")


def get(db, key):
    for i in range(len(key)):
        node = deserialize(db_get(db, key[:i]))
        if node is None:
            return None
        elif isinstance(node, LeafNode):
            return node.value
        elif isinstance(node, BranchNode):
            continue


def zpad32(x):
    return x + b'\x00' * (32 - len(x))


if __name__ == '__main__':
    db = leveldb.LevelDB(sys.argv[1])
    values = {
        b'cow': b'bovine',
        b'dog': b'canine',
        b'hippopotamus': b'riverhorse',
        b'hippogriff': b'harry',
    }
    hashes = [None]
    for k, v in values.items():
        add(db, zpad32(k), zpad32(v))
        hashes.append(hash_node(deserialize(db_get(db, b''))))
    print([hash_to_display_form(x) for x in hashes])
    print("Created full tree, hash: {}".format("0x"+hashes.pop().hex()))
    for k, v in values.items():
        assert get(db, zpad32(k)) == zpad32(v)
    for k, v in reversed(values.items()):
        delete(db, zpad32(k))
        assert hashes.pop() == hash_node(deserialize(db_get(db, b'')))
