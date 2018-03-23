from new_bintrie import b0, b1, KV_TYPE, BRANCH_TYPE, LEAF_TYPE, parse_node, encode_kv_node, encode_branch_node, encode_leaf_node
from bin_utils import encode_bin_path, decode_bin_path, common_prefix_length, encode_bin, decode_bin
from ethereum.utils import sha3 as _sha3, encode_hex

sha3_cache = {}

def sha3(x):
    if x not in sha3_cache:
        sha3_cache[x] = _sha3(x)
    return sha3_cache[x]

def quick_encode(nodes):
    o = b''
    for node in nodes:
        o += bytes([len(node) // 65536, len(node) // 256, len(node)]) + node
    return o

def quick_decode(nodedata):
    o = []
    pos = 0
    while pos < len(nodedata):
        L = nodedata[pos] * 65536 + nodedata[pos+1] * 256 + nodedata[pos+2]
        o.append(nodedata[pos+3: pos+3+L])
        pos += 3+L
    return o


class WrapperDB():
    def __init__(self, parent_db):
        self.parent_db = parent_db
        self.substores = {}
        self.node_to_substore = {}
        self.new_nodes = {}
        self.parent_db_reads = 0
        self.parent_db_writes = 0
        self.printing_mode = False

    # Loads a substore (RLP-encoded list of closeby trie nodes) from the DB
    def fetch_substore(self, key):
        substore_values = self.parent_db.get(key)
        assert substore_values is not None
        children = quick_decode(substore_values)
        self.parent_db_reads += 1
        self.substores[key] = {sha3(n): n for n in children}
        self.node_to_substore.update({sha3(n): key for n in children})
        assert key in self.node_to_substore and key in self.substores

    def get(self, k):
        if k in self.new_nodes:
            return self.new_nodes[k]
        if k not in self.node_to_substore:
            self.fetch_substore(k)
        o = self.substores[self.node_to_substore[k]][k]
        assert sha3(o) == k
        return o

    def put(self, k, v):
        if k not in self.new_nodes and k not in self.node_to_substore:
            self.new_nodes[k] = v

    # Given a key, returns a collection of candidate nodes to form
    # a substore, as well as the children of that substore
    def get_substore_candidate_and_children(self, key, depth=5):
        if depth == 0:
            return [], [key]
        elif self.parent_db.get(key) is not None:
            return [], [key]
        else:
            node = self.get(key)
            L, R, nodetype = parse_node(node)
            if nodetype == BRANCH_TYPE:
                Ln, Lc = self.get_substore_candidate_and_children(L, depth-1)
                Rn, Rc = self.get_substore_candidate_and_children(R, depth-1)
                return [node] + Ln + Rn, Lc + Rc
            elif nodetype == KV_TYPE:
                Rn, Rc = self.get_substore_candidate_and_children(R, depth-1)
                return [node] + Rn, Rc
            elif nodetype == LEAF_TYPE:
                return [node], []

    # Commits to the parent DB
    def commit(self):
        processed = {}
        assert_exists = {}
        for k, v in self.new_nodes.items():
            if k in processed:
                continue
            nodes, children = self.get_substore_candidate_and_children(k)
            if not nodes:
                continue
            assert k == sha3(nodes[0])
            for c in children:
                assert_exists[c] = True
                if c not in self.substores:
                    self.fetch_substore(c)
                cvalues = list(self.substores[c].values())
                if len(quick_encode(cvalues + nodes)) < 3072:
                    del self.substores[c]
                    nodes.extend(cvalues)
            self.parent_db.put(k, quick_encode(nodes))
            self.parent_db_writes += 1
            self.substores[k] = {}
            for n in nodes:
                h = sha3(n)
                self.substores[k][h] = n
                self.node_to_substore[h] = k
                processed[h] = k
            for c in assert_exists:
                assert self.parent_db.get(c) is not None
        print('reads', self.parent_db_reads, 'writes', self.parent_db_writes)
        self.parent_db_reads = self.parent_db_writes = 0
        self.new_nodes = {}

    def clear_cache(self):
        assert len(self.new_nodes) == 0
        self.substores = {}
        self.node_to_substore = {}
