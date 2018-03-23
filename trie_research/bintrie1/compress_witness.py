from ethereum.utils import sha3, encode_hex
from new_bintrie import parse_node, KV_TYPE, BRANCH_TYPE, LEAF_TYPE, encode_bin_path, encode_kv_node, encode_branch_node, decode_bin_path

KV_COMPRESS_TYPE = 128
BRANCH_LEFT_TYPE = 129
BRANCH_RIGHT_TYPE = 130

def compress(witness):
    parentmap = {}
    leaves = []
    for w in witness:
        L, R, nodetype = parse_node(w)
        if nodetype == LEAF_TYPE:
            leaves.append(w)
        elif nodetype == KV_TYPE:
            parentmap[R] = w
        elif nodetype == BRANCH_TYPE:
            parentmap[L] = w
            parentmap[R] = w
    used = {}
    proof = []
    for node in leaves:
        proof.append(node)
        used[node] = True
        h = sha3(node)
        while h in parentmap:
            node = parentmap[h]
            L, R, nodetype = parse_node(node)
            if nodetype == KV_TYPE:
                proof.append(bytes([KV_COMPRESS_TYPE]) + encode_bin_path(L))
            elif nodetype == BRANCH_TYPE and L == h:
                proof.append(bytes([BRANCH_LEFT_TYPE]) + R)
            elif nodetype == BRANCH_TYPE and R == h:
                proof.append(bytes([BRANCH_RIGHT_TYPE]) + L)
            else:
                raise Exception("something is wrong")
            h = sha3(node)
            if h in used:
                proof.pop()
                break
            used[h] = True
    assert len(used) == len(proof)
    return proof

# Input: a serialized node
def parse_proof_node(node):
    if node[0] == BRANCH_LEFT_TYPE:
        # Output: right child, node type
        return node[1:33], BRANCH_LEFT_TYPE
    elif node[0] == BRANCH_RIGHT_TYPE:
        # Output: left child, node type
        return node[1:33], BRANCH_RIGHT_TYPE
    elif node[0] == KV_COMPRESS_TYPE:
        # Output: keypath: child, node type
        return decode_bin_path(node[1:]), KV_COMPRESS_TYPE
    elif node[0] == LEAF_TYPE:
        # Output: None, value, node type
        return node[1:], LEAF_TYPE
    else:
        raise Exception("Bad node")

def expand(proof):
    witness = []
    lasthash = None
    for p in proof:
        sub, nodetype = parse_proof_node(p)
        if nodetype == LEAF_TYPE:
            witness.append(p)
            lasthash = sha3(p)
        elif nodetype == KV_COMPRESS_TYPE:
            fullnode = encode_kv_node(sub, lasthash)
            witness.append(fullnode)
            lasthash = sha3(fullnode)
        elif nodetype == BRANCH_LEFT_TYPE:
            fullnode = encode_branch_node(lasthash, sub)
            witness.append(fullnode)
            lasthash = sha3(fullnode)
        elif nodetype == BRANCH_RIGHT_TYPE:
            fullnode = encode_branch_node(sub, lasthash)
            witness.append(fullnode)
            lasthash = sha3(fullnode)
        else:
            raise Exception("Bad node")
    return witness
