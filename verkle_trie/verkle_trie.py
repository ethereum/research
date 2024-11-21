import pippenger
import blst
import hashlib
from random import randint, shuffle
from poly_utils import PrimeField
from time import time
from kzg_utils import KzgUtils
from fft import fft
import sys

#
# Proof of concept implementation for verkle tries
#
# All polynomials in this implementation are represented in evaluation form, i.e. by their values
# on DOMAIN. See kzg_utils.py for more explanation
#

# BLS12_381 curve modulus
MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

# Primitive root for the field
PRIMITIVE_ROOT = 7

assert pow(PRIMITIVE_ROOT, (MODULUS - 1) // 2, MODULUS) != 1
assert pow(PRIMITIVE_ROOT, MODULUS - 1, MODULUS) == 1

primefield = PrimeField(MODULUS)

# Verkle trie parameters
KEY_LENGTH = 256 # bits
WIDTH_BITS = 8
WIDTH = 2**WIDTH_BITS

ROOT_OF_UNITY = pow(PRIMITIVE_ROOT, (MODULUS - 1) // WIDTH, MODULUS)
DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(WIDTH)]

# Number of key-value pairs to insert
NUMBER_INITIAL_KEYS = 2**15

# Number of keys to insert after computing initial tree
NUMBER_ADDED_KEYS = 512

# Number of keys to delete
NUMBER_DELETED_KEYS = 512

# Number of key/values pair in proof
NUMBER_KEYS_PROOF = 5000

def generate_setup(size, secret):
    """
    Generates a setup in the G1 group and G2 group, as well as the Lagrange polynomials in G1 (via FFT)
    """
    g1_setup = [blst.G1().mult(pow(secret, i, MODULUS)) for i in range(size)]
    g2_setup = [blst.G2().mult(pow(secret, i, MODULUS)) for i in range(size)]
    g1_lagrange = fft(g1_setup, MODULUS, ROOT_OF_UNITY, inv=True)
    return {"g1": g1_setup, "g2": g2_setup, "g1_lagrange": g1_lagrange}


def get_verkle_indices(key):
    """
    Generates the list of verkle indices for key
    """
    x = int.from_bytes(key, "big")
    last_index_bits = KEY_LENGTH % WIDTH_BITS
    index = (x % (2**last_index_bits)) << (WIDTH_BITS - last_index_bits)
    x //= 2**last_index_bits
    indices = [index]
    for i in range((KEY_LENGTH - 1) // WIDTH_BITS):
        index = x % WIDTH
        x //= WIDTH
        indices.append(index)
    return tuple(reversed(indices))


def hash(x):
    if isinstance(x, bytes):
        return hashlib.sha256(x).digest()
    elif isinstance(x, blst.P1):
        return hash(x.compress())
    b = b""
    for a in x:
        if isinstance(a, bytes):
            b += a
        elif isinstance(a, int):
            b += a.to_bytes(32, "little")
        elif isinstance(a, blst.P1):
            b += hash(a.compress())
    return hash(b)


def hash_to_int(x):
    return int.from_bytes(hash(x), "little")


def insert_verkle_node(root, key, value):
    """
    Insert node without updating hashes/commitments (useful for building a full trie)
    """
    current_node = root
    indices = iter(get_verkle_indices(key))
    index = None
    while current_node["node_type"] == "inner":
        previous_node = current_node
        previous_index = index
        index = next(indices)
        if index in current_node:
            current_node = current_node[index]
        else:
            current_node[index] = {"node_type": "leaf", "key": key, "value": value}
            return
    if current_node["key"] == key:
        current_node["value"] = value
    else:
        previous_node[index] = {"node_type": "inner", "commitment": blst.G1().mult(0)}
        insert_verkle_node(root, key, value)
        insert_verkle_node(root, current_node["key"], current_node["value"])


def update_verkle_node(root, key, value):
    """
    Update or insert node and update all commitments and hashes
    """
    current_node = root
    indices = iter(get_verkle_indices(key))
    index = None
    path = []

    new_node = {"node_type": "leaf", "key": key, "value": value}
    add_node_hash(new_node)

    while True:
        index = next(indices)
        path.append((index, current_node))
        if index in current_node:
            if current_node[index]["node_type"] == "leaf":
                old_node = current_node[index]
                if current_node[index]["key"] == key:
                    current_node[index] = new_node
                    value_change = (MODULUS + int.from_bytes(new_node["hash"], "little")
                                    - int.from_bytes(old_node["hash"], "little")) % MODULUS
                    break
                else:
                    new_inner_node = {"node_type": "inner"}
                    new_index = next(indices)
                    old_index = get_verkle_indices(old_node["key"])[len(path)]
                    # TODO! Handle old_index == new_index
                    assert old_index != new_index
                    new_inner_node[new_index] = new_node
                    new_inner_node[old_index] = old_node
                    add_node_hash(new_inner_node)
                    current_node[index] = new_inner_node
                    value_change = (MODULUS + int.from_bytes(new_inner_node["hash"], "little")
                                    - int.from_bytes(old_node["hash"], "little")) % MODULUS
                    break
            current_node = current_node[index]
        else:
            current_node[index] = new_node
            value_change = int.from_bytes(new_node["hash"], "little") % MODULUS
            break
    
    # Update all the parent commitments along 'path'
    for index, node in reversed(path):
        node["commitment"].add(SETUP["g1_lagrange"][index].dup().mult(value_change))
        old_hash = node["hash"]
        new_hash = hash(node["commitment"])
        node["hash"] = new_hash
        value_change = (MODULUS + int.from_bytes(new_hash, "little")
                        - int.from_bytes(old_hash, "little")) % MODULUS


def get_only_child(node):
    """
    Returns the only child of a node which has only one child. Returns 'None' if node has 0 or >1 children
    """
    child_count = 0
    only_child = None
    for key in node:
        if isinstance(key, int):
            child_count += 1
            only_child = node[key]
    return only_child if child_count == 1 else None


def delete_verkle_node(root, key):
    """
    Delete node and update all commitments and hashes
    """
    current_node = root
    indices = iter(get_verkle_indices(key))
    index = None
    path = []

    while True:
        index = next(indices)
        path.append((index, current_node))
        assert index in current_node, "Tried to delete non-existent key"
        if current_node[index]["node_type"] == "leaf":
            deleted_node = current_node[index]
            assert deleted_node["key"] == key, "Tried to delete non-existent key"
            del current_node[index]
            value_change = (MODULUS - int.from_bytes(deleted_node["hash"], "little")) % MODULUS
            break
        current_node = current_node[index]
    
    # Update all the parent commitments along 'path'
    replacement_node = None
    for index, node in reversed(path):
        if replacement_node != None:
            node[index] = replacement_node
            replacement_node = None
        only_child = get_only_child(node)
        if only_child != None and only_child["node_type"] == "leaf" and node != root:
            replacement_node = only_child
            value_change = (MODULUS + int.from_bytes(only_child["hash"], "little")
                            - int.from_bytes(node["hash"], "little")) % MODULUS
        else:            
            node["commitment"].add(SETUP["g1_lagrange"][index].dup().mult(value_change))
            old_hash = node["hash"]
            new_hash = hash(node["commitment"])
            node["hash"] = new_hash
            value_change = (MODULUS + int.from_bytes(new_hash, "little")
                            - int.from_bytes(old_hash, "little")) % MODULUS


def add_node_hash(node):
    """
    Recursively adds all missing commitments and hashes to a verkle trie structure.
    """
    if node["node_type"] == "leaf":
        node["hash"] = hash([node["key"], node["value"]])
    if node["node_type"] == "inner":
        lagrange_polynomials = []
        values = {}
        for i in range(WIDTH):
            if i in node:
                if "hash" not in node[i]:
                    add_node_hash(node[i])
                values[i] = int.from_bytes(node[i]["hash"], "little")
        commitment = kzg_utils.compute_commitment_lagrange(values)
        node["commitment"] = commitment
        node["hash"] = hash(commitment.compress())


def get_total_depth(root):
    """
    Computes the total depth (sum of the depth of all nodes) of a verkle trie
    """
    if root["node_type"] == "inner":
        total_depth = 0
        num_nodes = 0
        for i in range(WIDTH):
            if i in root:
                depth, nodes = get_total_depth(root[i])
                num_nodes += nodes
                total_depth += nodes + depth
        return total_depth, num_nodes
    else:
        return 0, 1


def check_valid_tree(root, is_trie_root=True):
    """
    Checks that the tree is valid
    """
    if root["node_type"] == "inner":
        if not is_trie_root:
            only_child = get_only_child(root)
            if only_child is not None:
                assert only_child["node_type"] == "inner"
    
        lagrange_polynomials = []
        values = {}
        for i in range(WIDTH):
            if i in root:
                if "hash" not in root[i]:
                    add_node_hash(node[i])
                values[i] = int.from_bytes(root[i]["hash"], "little")
        commitment = kzg_utils.compute_commitment_lagrange(values)
        assert root["commitment"].is_equal(commitment)
        assert root["hash"] == hash(commitment.compress())

        for i in range(WIDTH):
            if i in root:
                check_valid_tree(root[i], False)
    else:
        assert root["hash"] == hash([root["key"], root["value"]])


def get_average_depth(trie):
    """
    Get the average depth of nodes in a verkle trie
    """
    depth, nodes = get_total_depth(trie)
    return depth / nodes


def find_node(root, key):
    """
    Finds 'key' in verkle trie. Returns the full node (not just the value) or None if not present
    """
    current_node = root
    indices = iter(get_verkle_indices(key))
    while current_node["node_type"] == "inner":
        index = next(indices)
        if index in current_node:
            current_node = current_node[index]
        else:
            return None
    if current_node["key"] == key:
        return current_node
    return None


def find_node_with_path(root, key):
    """
    As 'find_node', but returns the path of all nodes on the way to 'key' as well as their index
    """
    current_node = root
    indices = iter(get_verkle_indices(key))
    path = []
    current_index_path = []
    while current_node["node_type"] == "inner":
        index = next(indices)
        path.append((tuple(current_index_path), index, current_node))
        current_index_path.append(index)
        if index in current_node:
            current_node = current_node[index]
        else:
            return path, None
    if current_node["key"] == key:
        return path, current_node
    return path, None
    

def get_proof_size(proof):
    depths, commitments_sorted_by_index_serialized, D_serialized, y, sigma_serialized = proof
    size = len(depths) # assume 8 bit integer to represent the depth
    size += 48 * len(commitments_sorted_by_index_serialized)
    size += 48 + 32 + 48
    return size

lasttime = [0]


def start_logging_time_if_eligible(string, eligible):
    if eligible:
        print(string, file=sys.stderr)
        lasttime[0] = time()

        
def log_time_if_eligible(string, width, eligible):
    if eligible:
        print(string + ' ' * max(1, width - len(string)) + "{0:7.3f} s".format(time() - lasttime[0]), file=sys.stderr)
        lasttime[0] = time()


def make_kzg_multiproof(Cs, fs, indices, ys, display_times=True):
    """
    Computes a KZG multiproof according to the schema described here:
    https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html

    zs[i] = DOMAIN[indexes[i]]
    """

    # Step 1: Construct g(X) polynomial in evaluation form
    r = hash_to_int([hash(C) for C in Cs] + ys + [kzg_utils.DOMAIN[i] for i in indices]) % MODULUS

    log_time_if_eligible("   Hashed to r", 30, display_times)

    g = [0 for i in range(WIDTH)]
    power_of_r = 1
    for f, index in zip(fs, indices):
        quotient = kzg_utils.compute_inner_quotient_in_evaluation_form(f, index)
        for i in range(WIDTH):
            g[i] += power_of_r * quotient[i]

        power_of_r = power_of_r * r % MODULUS

    log_time_if_eligible("   Computed g polynomial", 30, display_times)

    D = kzg_utils.compute_commitment_lagrange({i: v for i, v in enumerate(g)})

    log_time_if_eligible("   Computed commitment D", 30, display_times)

    # Step 2: Compute h in evaluation form
    
    t = hash_to_int([r, D]) % MODULUS
    
    h = [0 for i in range(WIDTH)]
    power_of_r = 1
    
    for f, index in zip(fs, indices):
        denominator_inv = primefield.inv(t - DOMAIN[index])
        for i in range(WIDTH):
            h[i] += power_of_r * f[i] * denominator_inv % MODULUS
            
        power_of_r = power_of_r * r % MODULUS
   
    log_time_if_eligible("   Computed h polynomial", 30, display_times)

    # Step 3: Evaluate and compute KZG proofs

    y, pi = kzg_utils.evaluate_and_compute_kzg_proof(h, t)
    w, rho = kzg_utils.evaluate_and_compute_kzg_proof(g, t)


    # Compress both proofs into one

    E = kzg_utils.compute_commitment_lagrange({i: v for i, v in enumerate(h)})
    q = hash_to_int([E, D, y, w])
    sigma = pi.dup().add(rho.dup().mult(q))

    log_time_if_eligible("   Computed KZG proofs", 30, display_times)

    return D.compress(), y, sigma.compress()


def check_kzg_multiproof(Cs, indices, ys, proof, display_times=True):
    """
    Verifies a KZG multiproof according to the schema described here:
    https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html
    """

    D_serialized, y, sigma_serialized = proof
    D = blst.P1(D_serialized)
    sigma = blst.P1(sigma_serialized)

    # Step 1
    r = hash_to_int([hash(C) for C in Cs] + ys + [kzg_utils.DOMAIN[i] for i in indices]) % MODULUS

    log_time_if_eligible("   Computed r hash", 30, display_times)
    
    # Step 2
    t = hash_to_int([r, D])
    E_coefficients = []
    g_2_of_t = 0
    power_of_r = 1

    for index, y in zip(indices, ys):
        E_coefficient = primefield.div(power_of_r, t - DOMAIN[index])
        E_coefficients.append(E_coefficient)
        g_2_of_t += E_coefficient * y % MODULUS
            
        power_of_r = power_of_r * r % MODULUS

    log_time_if_eligible("   Computed g2 and e coeffs", 30, display_times)
    
    E = pippenger.pippenger_simple(Cs, E_coefficients)

    log_time_if_eligible("   Computed E commitment", 30, display_times)

    # Step 3 (Check KZG proofs)
    w = (y - g_2_of_t) % MODULUS

    q = hash_to_int([E, D, y, w])

    if not kzg_utils.check_kzg_proof(E.dup().add(D.dup().mult(q)), t, y + q * w, sigma):
        return False

    log_time_if_eligible("   Checked KZG proofs", 30, display_times)

    return True


def make_verkle_proof(trie, keys, display_times=True):
    """
    Creates a proof for the 'keys' in the verkle trie given by 'trie'
    """

    start_logging_time_if_eligible("   Starting proof computation", display_times)

    # Step 0: Find all keys in the trie
    nodes_by_index = {}
    nodes_by_index_and_subindex = {}
    values = []
    depths = []
    for key in keys:
        path, node = find_node_with_path(trie, key)
        depths.append(len(path))
        values.append(node["value"])
        for index, subindex, node in path:
            nodes_by_index[index] = node
            nodes_by_index_and_subindex[(index, subindex)] = node

    log_time_if_eligible("   Computed key paths", 30, display_times)
    
    # All commitments, but without any duplications. These are for sending over the wire as part of the proof
    nodes_sorted_by_index = list(map(lambda x: x[1], sorted(nodes_by_index.items())))
    
    # Nodes sorted 
    nodes_sorted_by_index_and_subindex = list(map(lambda x: x[1], sorted(nodes_by_index_and_subindex.items())))
    
    indices = list(map(lambda x: x[0][1], sorted(nodes_by_index_and_subindex.items())))
    
    ys = list(map(lambda x: int.from_bytes(x[1][x[0][1]]["hash"], "little"), sorted(nodes_by_index_and_subindex.items())))
    
    log_time_if_eligible("   Sorted all commitments", 30, display_times)

    fs = []
    Cs = [x["commitment"] for x in nodes_sorted_by_index_and_subindex]

    for node in nodes_sorted_by_index_and_subindex:
        fs.append([int.from_bytes(node[i]["hash"], "little") if i in node else 0 for i in range(WIDTH)])

    D, y, sigma = make_kzg_multiproof(Cs, fs, indices, ys, display_times)

    commitments_sorted_by_index_serialized = [x["commitment"].compress() for x in nodes_sorted_by_index[1:]]
    
    log_time_if_eligible("   Serialized commitments", 30, display_times)

    return depths, commitments_sorted_by_index_serialized, D, y, sigma


def check_verkle_proof(trie, keys, values, proof, display_times=True):
    """
    Checks Verkle tree proof according to
    https://notes.ethereum.org/nrQqhVpQRi6acQckwm1Ryg?both
    """

    start_logging_time_if_eligible("   Starting proof check", display_times)

    # Unpack the proof
    depths, commitments_sorted_by_index_serialized, D_serialized, y, sigma_serialized = proof
    commitments_sorted_by_index = [blst.P1(trie)] + [blst.P1(x) for x in commitments_sorted_by_index_serialized]

    all_indices = set()
    all_indices_and_subindices = set()
    
    leaf_values_by_index_and_subindex = {}

    # Find all required indices
    for key, value, depth in zip(keys, values, depths):
        verkle_indices = get_verkle_indices(key)
        for i in range(depth):
            all_indices.add(verkle_indices[:i])
            all_indices_and_subindices.add((verkle_indices[:i], verkle_indices[i]))
        leaf_values_by_index_and_subindex[(verkle_indices[:depth - 1], verkle_indices[depth - 1])] = hash([key, value])
    
    all_indices = sorted(all_indices)
    all_indices_and_subindices = sorted(all_indices_and_subindices)

    log_time_if_eligible("   Computed indices", 30, display_times)

    # Step 0: recreate the commitment list sorted by indices
    commitments_by_index = {index: commitment for index, commitment in zip(all_indices, commitments_sorted_by_index)}
    commitments_by_index_and_subindex = {index_and_subindex: commitments_by_index[index_and_subindex[0]]
                                            for index_and_subindex in all_indices_and_subindices}

    subhashes_by_index_and_subindex = {}
    for index_and_subindex in all_indices_and_subindices:
        full_subindex = index_and_subindex[0] + (index_and_subindex[1],)
        if full_subindex in commitments_by_index:
            subhashes_by_index_and_subindex[index_and_subindex] = hash(commitments_by_index[full_subindex])
        else:
            subhashes_by_index_and_subindex[index_and_subindex] = leaf_values_by_index_and_subindex[index_and_subindex]
    
    Cs = list(map(lambda x: x[1], sorted(commitments_by_index_and_subindex.items())))
    
    indices = list(map(lambda x: x[1], sorted(all_indices_and_subindices)))
    
    ys = list(map(lambda x: int.from_bytes(x[1], "little"), sorted(subhashes_by_index_and_subindex.items())))

    log_time_if_eligible("   Recreated commitment lists", 30, display_times)

    return check_kzg_multiproof(Cs, indices, ys, [D_serialized, y, sigma_serialized], display_times)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        WIDTH_BITS = int(sys.argv[1])
        WIDTH = 2 ** WIDTH_BITS
        ROOT_OF_UNITY = pow(PRIMITIVE_ROOT, (MODULUS - 1) // WIDTH, MODULUS)
        DOMAIN = [pow(ROOT_OF_UNITY, i, MODULUS) for i in range(WIDTH)]

        NUMBER_INITIAL_KEYS = int(sys.argv[2])

        NUMBER_KEYS_PROOF = int(sys.argv[3])

        NUMBER_DELETED_KEYS = 0
        NUMBER_ADDED_KEYS = 0
    
    SETUP = generate_setup(WIDTH, 8927347823478352432985)
    kzg_utils = KzgUtils(MODULUS, WIDTH, DOMAIN, SETUP, primefield)


    # Build a random verkle trie
    root = {"node_type": "inner", "commitment": blst.G1().mult(0)}

    values = {}

    for i in range(NUMBER_INITIAL_KEYS):
        key = randint(0, 2**256-1).to_bytes(32, "little")
        value = randint(0, 2**256-1).to_bytes(32, "little")
        insert_verkle_node(root, key, value)
        values[key] = value
    
    average_depth = get_average_depth(root)
        
    print("Inserted {0} elements for an average depth of {1:.3f}".format(NUMBER_INITIAL_KEYS, average_depth), file=sys.stderr)

    time_a = time()
    add_node_hash(root)
    time_b = time()

    print("Computed verkle root in {0:.3f} s".format(time_b - time_a), file=sys.stderr)

    if NUMBER_ADDED_KEYS > 0:

        time_a = time()
        check_valid_tree(root)
        time_b = time()
        
        print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)

        time_x = time()
        for i in range(NUMBER_ADDED_KEYS):
            key = randint(0, 2**256-1).to_bytes(32, "little")
            value = randint(0, 2**256-1).to_bytes(32, "little")
            update_verkle_node(root, key, value)
            values[key] = value
        time_y = time()
            
        print("Additionally inserted {0} elements in {1:.3f} s".format(NUMBER_ADDED_KEYS, time_y - time_x), file=sys.stderr)
        print("Keys in tree now: {0}, average depth: {1:.3f}".format(get_total_depth(root)[1], get_average_depth(root)), file=sys.stderr)

        time_a = time()
        check_valid_tree(root)
        time_b = time()
        
        print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)
    
    if NUMBER_DELETED_KEYS > 0:

        all_keys = list(values.keys())
        shuffle(all_keys)

        keys_to_delete = all_keys[:NUMBER_DELETED_KEYS]

        time_a = time()
        for key in keys_to_delete:
            delete_verkle_node(root, key)
            del values[key]
        time_b = time()
        
        print("Deleted {0} elements in {1:.3f} s".format(NUMBER_DELETED_KEYS, time_b - time_a), file=sys.stderr)
        print("Keys in tree now: {0}, average depth: {1:.3f}".format(get_total_depth(root)[1], get_average_depth(root)), file=sys.stderr)


        time_a = time()
        check_valid_tree(root)
        time_b = time()
        
        print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)


    all_keys = list(values.keys())
    shuffle(all_keys)

    keys_in_proof = all_keys[:NUMBER_KEYS_PROOF]

    time_a = time()
    proof = make_verkle_proof(root, keys_in_proof)
    time_b = time()
    
    proof_size = get_proof_size(proof)
    proof_time = time_b - time_a
    
    print("Computed proof for {0} keys (size = {1} bytes) in {2:.3f} s".format(NUMBER_KEYS_PROOF, proof_size, time_b - time_a), file=sys.stderr)

    time_a = time()
    check_verkle_proof(root["commitment"].compress(), keys_in_proof, [values[key] for key in keys_in_proof], proof)
    time_b = time()
    check_time = time_b - time_a

    print("Checked proof in {0:.3f} s".format(time_b - time_a), file=sys.stderr)

    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(WIDTH_BITS, WIDTH, NUMBER_INITIAL_KEYS, NUMBER_KEYS_PROOF, average_depth, proof_size, proof_time, check_time))
