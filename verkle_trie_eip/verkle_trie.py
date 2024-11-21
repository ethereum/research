from bandersnatch import Point, Scalar
import hashlib
from random import randint, shuffle, choice
from poly_utils import PrimeField
from time import time
from ipa_utils import IPAUtils, hash
import sys

#
# Proof of concept implementation for verkle tries
#
# All polynomials in this implementation are represented in evaluation form, i.e. by their values
# on the domain 0, 1, ..., WIDTH - 1
#
# Ethereum-specific implementation according to this EIP draft:
# https://notes.ethereum.org/@vbuterin/verkle_tree_eip
#

# Bandersnatch curve modulus
MODULUS = 13108968793781547619861935127046491459309155893440570251786403306729687672801

# Verkle trie parameters
KEY_LENGTH = 256 # bits
WIDTH_BITS = 8
WIDTH = 2**WIDTH_BITS

primefield = PrimeField(MODULUS, WIDTH)

# Number of key-value pairs to insert
NUMBER_STEMS = 2**15
CHUNKS_PER_STEM = 10
# Needs to be less than WIDTH * NUMBER_STEMS
NUMBER_CHUNKS = CHUNKS_PER_STEM * NUMBER_STEMS

# Number of extra stems to add to tree
NUMBER_ADDED_STEMS = 512

# Number of chunks to add to existing stems
NUMBER_ADDED_CHUNKS = 512

# Number of actually existing key/values pair in proof
NUMBER_EXISTING_KEYS_PROOF = 5000

# Added stems and chunks randomly (most likely empty)
NUMBER_RANDOM_STEMS_PROOF = 1000
NUMBER_RANDOM_CHUNKS_PROOF = 1000

NUMBER_VALUES_UPDATED = 1000

# Verkle trie constants
VERKLE_TRIE_NODE_TYPE_INNER = 0
VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE = 1

# Verkle proof wire constants
VERKLE_PROOF_COMMITMENT_TYPE_INNER = 0
VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION = 1
VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1 = 2
VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C2 = 3

VERKLE_PROOF_EXTENSION_PRESENT_NOEXTENSION = 0
VERKLE_PROOF_EXTENSION_PRESENT_PRESENT = 1
VERKLE_PROOF_EXTENSION_PRESENT_OTHERSTEM = 2 # Used to indicate that there is an extension present, but for a different stem

def generate_basis(size):
    """
    Generates a basis for Pedersen commitments
    """
    # TODO: Currently random points that differ on every run.
    # Implement reproducable basis generation once hash_to_curve is provided
    BASIS_G = [Point(generator=False) for i in range(WIDTH)]
    BASIS_Q = Point(generator=False)
    return {"G": BASIS_G, "Q": BASIS_Q}


def get_stem(key):
    return key[:31]


def get_suffix(key):
    return key[31]


def commitment_to_field(commitment):
    return int.from_bytes(commitment.serialize(), "little") % MODULUS


# Node types: (Represented as python dicts)
#
# VERKLE_TRIE_NODE_TYPE_INNER:
#   0-255 refs to child node
#   "commitment": commitment
#   "commitment_field": commitment % MODULUS
#
# VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE:
#   0-255 values (as bytes)
#   "stem": stem (31 bytes)
#   "C1": C1
#   "C1_field": C1 % MODULUS
#   "C2": C2
#   "C2_field": C2 % MODULUS
#   "commitment": commitment
#   "commitment_field": commitment % MODULUS


def update_verkle_tree_nocommitmentupdate(root_node, key, value):
    """
    Insert node without updating commitments (useful for building a full trie and adding commitments at the end)
    """

    current_node = root_node
    stem = get_stem(key)
    suffix = get_suffix(key)
    index = None
    depth = 0

    while current_node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
        previous_node = current_node
        index = stem[depth]
        depth += 1
        if index in current_node:
            current_node = current_node[index]
        else:
            current_node[index] = {"node_type": VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE, "stem": stem, suffix: value}
            return

    if current_node["stem"] == stem:
        current_node[suffix] = value
    else:
        old_suffix_tree = current_node
        old_stem = old_suffix_tree["stem"]

        new_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
        previous_node[index] = new_inner_node
        previous_node = new_inner_node
        
        while old_stem[depth] == stem[depth]:
            index = stem[depth]
            new_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
            previous_node[index] = new_inner_node
            previous_node = new_inner_node
            depth += 1

        new_inner_node[stem[depth]] = {"node_type": VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE, "stem": stem, suffix: value}
        new_inner_node[old_stem[depth]] = old_suffix_tree


def update_verkle_tree(root_node, key, value):
    """
    Update or insert node and update all commitments
    """
    current_node = root_node
    index = None
    path = []
    stem = get_stem(key)
    suffix = get_suffix(key)

    while True:
        index = stem[len(path)]
        path.append((index, current_node))
        if index in current_node:
            if current_node[index]["node_type"] == VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE:
                old_node = current_node[index]
                if current_node[index]["stem"] == stem:
                    current_node = current_node[index]
                    old_value_lower = (int.from_bytes(current_node[suffix][:16], "little") + 2**128) if suffix in current_node else 0
                    old_value_upper = (int.from_bytes(current_node[suffix][16:], "little")) if suffix in current_node else 0
                    current_node[suffix] = value
                    new_value_lower = int.from_bytes(value[:16], "little") + 2**128
                    new_value_upper = int.from_bytes(value[16:], "little")
                    commitment_change = BASIS["G"][2 * suffix % 256].dup().mul((MODULUS + new_value_lower - old_value_lower) % MODULUS) \
                                        .add(BASIS["G"][(2 * suffix + 1) % 256].dup().mul((MODULUS + new_value_upper - old_value_upper) % MODULUS))
                    
                    if suffix < 128:
                        current_node["C1"].add(commitment_change)
                        new_field = commitment_to_field(current_node["C1"])
                        current_node["commitment"].add(BASIS["G"][2].dup().mul((MODULUS + new_field - current_node["C1_field"]) % MODULUS))
                        current_node["C1_field"] = new_field
                    else:
                        current_node["C2"].add(commitment_change)
                        new_field = commitment_to_field(current_node["C2"])
                        current_node["commitment"].add(BASIS["G"][3].dup().mul((MODULUS + new_field - current_node["C2_field"]) % MODULUS))
                        current_node["C2_field"] = new_field
                    new_field = commitment_to_field(current_node["commitment"])
                    value_change = (MODULUS + new_field - current_node["commitment_field"]) % MODULUS
                    current_node["commitment_field"] = new_field
                    break
                else:
                    new_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
                    new_index = stem[len(path)]
                    old_index = old_node["stem"][len(path)]
                    current_node[index] = new_inner_node

                    inserted_path = []
                    current_node = new_inner_node
                    while old_index == new_index:
                        index = new_index
                        next_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
                        current_node[index] = next_inner_node
                        inserted_path.append((index, current_node))
                        new_index = stem[len(path) + len(inserted_path)]
                        old_index = old_node["stem"][len(path) + len(inserted_path)]
                        current_node = next_inner_node

                    current_node[new_index] = {"node_type": VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE, "stem": stem, suffix: value}
                    current_node[old_index] = old_node

                    verkle_add_missing_commitments(current_node)
                    for index, node in reversed(inserted_path):
                        verkle_add_missing_commitments(node)

                    value_change = (MODULUS + new_inner_node["commitment_field"] - old_node["commitment_field"]) % MODULUS
                    break

            current_node = current_node[index]
        else:
            current_node[index] = {"node_type": VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE, "stem": stem, suffix: value}
            verkle_add_missing_commitments(current_node[index])
            value_change = current_node[index]["commitment_field"]
            break
    
    # Update all the ancestor commitments along `path`
    for index, node in reversed(path):
        node["commitment"].add(BASIS["G"][index].dup().mul(value_change))
        old_field = node["commitment_field"]
        new_field = commitment_to_field(node["commitment"])
        node["commitment_field"] = new_field
        value_change = (MODULUS + new_field - old_field) % MODULUS


def verkle_add_missing_commitments(node):
    """
    Recursively adds all missing commitments and hashes to a verkle trie structure.
    """
    if node["node_type"] == VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE:

        C1 = ipa_utils.pedersen_commit_sparse({2 * i + j: int.from_bytes(node[i][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                for i in range(128)
                                                for j in range(2)
                                                if i in node})

        C2 = ipa_utils.pedersen_commit_sparse({2 * i + j: int.from_bytes(node[128 + i][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                for i in range(128)
                                                for j in range(2)
                                                if 128 + i in node})

        C1_field = commitment_to_field(C1)
        C2_field = commitment_to_field(C2)

        node["C1"] = C1
        node["C1_field"] = C1_field
        node["C2"] = C2
        node["C2_field"] = C2_field

        commitment = ipa_utils.pedersen_commit_sparse({0: 1, 
                                                       1: int.from_bytes(node["stem"], "little"),
                                                       2: C1_field, 
                                                       3: C2_field})

        node["commitment"] = commitment
        node["commitment_field"] = commitment_to_field(commitment)

    elif node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
        child = {}
        for i in range(WIDTH):
            if i in node:
                if "commitment_field" not in node[i]:
                    verkle_add_missing_commitments(node[i])
                child[i] = node[i]["commitment_field"]
        commitment = ipa_utils.pedersen_commit_sparse(child)
        node["commitment"] = commitment
        node["commitment_field"] = int.from_bytes(commitment.serialize(), "little") % MODULUS


def check_valid_tree(node, prefix=b""):
    """
    Checks that the subtree starting at `node` with prefix `prefix` is valid.
    Returns all the values found in the tree as a dict.
    """
    values = {}
    if node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:

        child = {}
        for i in range(WIDTH):
            if i in node:
                child[i] = node[i]["commitment_field"]
        commitment = ipa_utils.pedersen_commit_sparse(child)
        assert node["commitment"] == commitment
        assert node["commitment_field"] == int.from_bytes(commitment.serialize(), "little") % MODULUS

        for i in range(WIDTH):
            if i in node:
                values.update(check_valid_tree(node[i], prefix + bytes([i])))
    else:
        assert node["node_type"] == VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE
        assert node["stem"][:len(prefix)] == prefix
        C1 = ipa_utils.pedersen_commit_sparse({2 * i + j: int.from_bytes(node[i][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                for i in range(128)
                                                for j in range(2)
                                                if i in node})

        C2 = ipa_utils.pedersen_commit_sparse({2 * i + j: int.from_bytes(node[128 + i][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                for i in range(128)
                                                for j in range(2)
                                                if 128 + i in node})

        C1_field = commitment_to_field(C1)
        C2_field = commitment_to_field(C2)

        assert node["C1"] == C1
        assert node["C1_field"] == C1_field
        assert node["C2"] == C2
        assert node["C2_field"] == C2_field

        commitment = ipa_utils.pedersen_commit_sparse({0: 1, 
                                                       1: int.from_bytes(node["stem"], "little"),
                                                       2: C1_field, 
                                                       3: C2_field})

        assert node["commitment"] == commitment
        assert node["commitment_field"] == commitment_to_field(commitment)

        for i in range(WIDTH):
            if i in node:
                values[node["stem"] + bytes([i])] = node[i]
    
    return values


def get_total_depth(node):
    """
    Computes the total depth (sum of the depth of all nodes) of a verkle trie as well as the total number of nonzero values.
    Depth refers to the total number of commitments on the path to revealing the values, including the root and the
    extension and suffix tree node. (So a single value in an otherwise empty verkle tree has depth 3:
    root + extension + suffix_tree)
    """
    if node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
        total_depth = 0
        num_nodes = 0
        for i in range(WIDTH):
            if i in node:
                depth, nodes = get_total_depth(node[i])
                num_nodes += nodes
                total_depth += nodes + depth
        return total_depth, num_nodes
    else:
        num_chunks = len([i for i  in range(256) if i in node])
        return num_chunks * 2, num_chunks


def get_average_depth(root_node):
    """
    Get the average depth of nodes in a verkle trie
    """
    depth, nodes = get_total_depth(root_node)
    return depth / nodes


def find_key_with_path(root_node, key):
    """
    Returns the path of all nodes on the way to `key` as well as their index
    """
    current_node = root_node
    node_path = []
    stem = get_stem(key)
    while current_node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
        index = key[len(node_path)]
        node_path.append((stem[:len(node_path)], index, current_node))
        if index in current_node:
            current_node = current_node[index]
        else:
            return node_path, None

    suffix = get_suffix(key)
    node_path.append((stem[:len(node_path)], suffix, current_node))

    if current_node["stem"] == stem and suffix in current_node:
        return node_path, current_node[suffix]

    return node_path, None
    

def get_proof_size(proof):
    depths, extension_present, commitments_sorted_by_path_serialized, other_stems, D_serialized, ipa_proof = proof
    size = len(depths) # assume 8 bit integer to represent the depth (5 bit) and extension_present(2 bit)
    size += 32 * len(commitments_sorted_by_path_serialized)
    size += 32 * len(other_stems)
    size += 32 + (len(ipa_proof) - 1) * 2 * 32 + 32
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


def make_ipa_multiproof(Cs, fs, zs, ys, display_times=True):
    """
    Computes an IPA multiproof according to the schema described here:
    https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html

    This proof makes the assumption that the domain is the integers 0, 1, 2, ... WIDTH - 1
    """

    # Step 1: Construct g(X) polynomial in evaluation form
    r = ipa_utils.hash_to_field(Cs + zs + ys) % MODULUS

    log_time_if_eligible("   Hashed to r", 30, display_times)

    g = [0 for i in range(WIDTH)]
    power_of_r = 1
    for f, index in zip(fs, zs):
        quotient = primefield.compute_inner_quotient_in_evaluation_form(f, index)
        for i in range(WIDTH):
            g[i] += power_of_r * quotient[i]

        power_of_r = power_of_r * r % MODULUS
    
    for i in range(len(g)):
        g[i] %= MODULUS

    log_time_if_eligible("   Computed g polynomial", 30, display_times)

    D = ipa_utils.pedersen_commit(g)

    log_time_if_eligible("   Computed commitment D", 30, display_times)

    # Step 2: Compute h in evaluation form
    
    t = ipa_utils.hash_to_field([r, D]) % MODULUS
    
    h = [0 for i in range(WIDTH)]
    power_of_r = 1
    
    for f, index in zip(fs, zs):
        denominator_inv = primefield.inv(t - primefield.DOMAIN[index])
        for i in range(WIDTH):
            h[i] += power_of_r * f[i] * denominator_inv % MODULUS
            
        power_of_r = power_of_r * r % MODULUS
   
    for i in range(len(h)):
        h[i] %= MODULUS

    log_time_if_eligible("   Computed h polynomial", 30, display_times)

    h_minus_g = [(h[i] - g[i]) % primefield.MODULUS for i in range(WIDTH)]

    # Step 3: Evaluate and compute IPA proofs

    E = ipa_utils.pedersen_commit(h)

    y, ipa_proof = ipa_utils.evaluate_and_compute_ipa_proof(E.dup().add(D.dup().mul(MODULUS-1)), h_minus_g, t)

    log_time_if_eligible("   Computed IPA proof", 30, display_times)

    return D.serialize(), ipa_proof


def check_ipa_multiproof(Cs, zs, ys, proof, display_times=True):
    """
    Verifies an IPA multiproof according to the schema described here:
    https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html
    """

    D_serialized, ipa_proof = proof

    D = Point().deserialize(D_serialized)

    # Step 1
    r = ipa_utils.hash_to_field(Cs + zs + ys)

    log_time_if_eligible("   Computed r hash", 30, display_times)
    
    # Step 2
    t = ipa_utils.hash_to_field([r, D])
    E_coefficients = {}
    g_2_of_t = 0
    power_of_r = 1

    C_by_serialized = {}

    for C, z, y in zip(Cs, zs, ys):
        E_coefficient = primefield.div(power_of_r, t - primefield.DOMAIN[z])
        C_serialized = C.serialize()
        C_by_serialized[C_serialized] = C
        E_coefficients[C_serialized] = E_coefficient if C_serialized not in E_coefficients \
                                        else (E_coefficients[C_serialized] + E_coefficient) % MODULUS
        g_2_of_t += E_coefficient * y % MODULUS

        power_of_r = power_of_r * r % MODULUS

    log_time_if_eligible("   Computed g2 and e coeffs", 30, display_times)
    
    E = Point().msm([C_by_serialized[x] for x in E_coefficients.keys()], E_coefficients.values())

    log_time_if_eligible("   Computed E commitment", 30, display_times)

    # Step 3 (Check IPA proofs)
    y = g_2_of_t % primefield.MODULUS

    if not ipa_utils.check_ipa_proof(E.dup().add(D.dup().mul(MODULUS - 1)), t, y, ipa_proof):
        return False

    log_time_if_eligible("   Checked IPA proof", 30, display_times)

    return True


def make_verkle_proof(root_node, keys, display_times=True):
    """
    Creates a proof for the `keys` in the verkle trie given by `root_node`

    This includes proving that a value is not in the verkle trie
    """

    start_logging_time_if_eligible("   Starting proof computation", display_times)

    #
    # Revealing a full verkle proof requires the following proofs for each `key`
    #
    # - all nodes on the path to the extension node, with z determined by stem             [VERKLE_PROOF_COMMITMENT_TYPE_INNER]
    # 
    # If there is an extension at the last place, then further it requires an extension proof:
    #
    # - the extension node, z = 0 (value = 1)                                              [VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION]
    # - the extension node, z = 1 (stem)                                                   [VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION]
    #
    # The stem of the extension can be a different stem, which has to be provided (either it is already in another key or it has to be added
    # to the proof)
    # If the stem is the stem of `key`, then the suffix tree for the suffix has to be revealed:
    #
    # - the extension node, z = 2/3 for C1/C2 (if suffix <128/>=128)                       [VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION]
    # - the suffix tree node C1/C2, z = 2 * suffix % 128     (value_lower + 2**128)        [VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C{1/2}]
    # - the suffix tree node C1/C2, z = 2 * suffix + 1 % 128 (value_upper)                 [VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C{1_2}]
    #

    # Step 0: Find all keys in the trie

    # Nodes by path -- path refers to the full path of the commitment in the verkle trie
    nodes_by_path = {}

    # Nodes by path and z -- z is the position of the node that is opened; note that the same node can be opened at several z
    nodes_by_path_and_z = {}

    # All values in order of keys. `None` is used for never written values and b"\0" * 32 for zero/deleted values
    values = []

    # Depth at which the extension node for the stem was found
    depths_by_stem = {}

    # Whether or not a given stem had an extension node or not
    extension_present_by_stem = {}

    # All the stems that are already part of the proof
    key_stems = set(get_stem(key) for key in keys)

    # In some cases, a key that is not present will end up in an extension node for another stem. In this case,
    # we need to reveal that other stem as part of the proof so that the verifier can check that the extension
    # node is indeed for the other stem.
    other_stems = set()

    for key in keys:
        node_path, value = find_key_with_path(root_node, key)
        stem = get_stem(key)
        suffix = get_suffix(key)

        values.append(value)

        for path, z, node in node_path:
            if node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
                nodes_by_path[path] = (VERKLE_PROOF_COMMITMENT_TYPE_INNER, node)
                nodes_by_path_and_z[(path, z)] = (VERKLE_PROOF_COMMITMENT_TYPE_INNER, node)

        node = node_path[-1][2]
        path = node_path[-1][0]
        if node["node_type"] == VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE:
            
            other_stem = node["stem"]
            depths_by_stem[stem] = len(node_path) - 1
            nodes_by_path[path] = (VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION, node)
            nodes_by_path_and_z[(path, 0)] = (VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION, node) # 1
            nodes_by_path_and_z[(path, 1)] = (VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION, node) # stem

            if other_stem == stem:
                extension_present_by_stem[stem] = VERKLE_PROOF_EXTENSION_PRESENT_PRESENT

                nodes_by_path_and_z[(path, 2 + suffix // 128)] = (VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION, node)  # C1/C2

                suffix_tree_path = path + bytes([2 if suffix < 128 else 3])
                suffix_tree_commitment = VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1 if suffix < 128 \
                                        else VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C2
                nodes_by_path[suffix_tree_path] = (suffix_tree_commitment, node)
                nodes_by_path_and_z[(suffix_tree_path, suffix * 2 % 256)] = (suffix_tree_commitment, node)       # value_lower
                nodes_by_path_and_z[(suffix_tree_path, (suffix * 2 + 1) % 256)] = (suffix_tree_commitment, node) # value_upper
            else:
                extension_present_by_stem[stem] = VERKLE_PROOF_EXTENSION_PRESENT_OTHERSTEM
                other_stems.add(other_stem)
        else:
            depths_by_stem[stem] = len(node_path)
            extension_present_by_stem[stem] = VERKLE_PROOF_EXTENSION_PRESENT_NOEXTENSION

    depths = list(map(lambda x: x[1], sorted(depths_by_stem.items())))
    extension_present = list(map(lambda x: x[1], sorted(extension_present_by_stem.items())))
        
    log_time_if_eligible("   Computed key paths", 30, display_times)
    
    # Nodes sorted 
    nodes_sorted_by_path_and_z = sorted(nodes_by_path_and_z.items())
    
    log_time_if_eligible("   Sorted all commitments", 30, display_times)
    
    zs = []
    ys = []
    fs = []
    Cs = []

    for path_and_z, node_type_and_node in nodes_sorted_by_path_and_z:
        path, z = path_and_z
        node_type, node = node_type_and_node
        zs.append(z)
        if node_type == VERKLE_PROOF_COMMITMENT_TYPE_INNER:
            Cs.append(node["commitment"])
            ys.append(node[z]["commitment_field"] if z in node else 0)
            fs.append([node[i]["commitment_field"] if i in node else 0 for i in range(WIDTH)])
        elif node_type == VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION:
            Cs.append(node["commitment"])
            if z == 0:
                ys.append(1)
            elif z == 1:
                ys.append(int.from_bytes(node["stem"], "little"))
            elif z == 2:
                ys.append(node["C1_field"])
            elif z == 3:
                ys.append(node["C2_field"])
            fs.append([1, 
                       int.from_bytes(node["stem"], "little"),
                       node["C1_field"],
                       node["C2_field"]]
                        + [0] * 252)
        elif node_type in [VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1, VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C2]:
            if node_type == VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1:
                Cs.append(node["C1"])
                suffix_offset = 0
            else:
                Cs.append(node["C2"])
                suffix_offset = 128
            suffix = suffix_offset + z // 2
            if suffix not in node:
                ys.append(0)
            else:
                if z % 2 == 0:
                    ys.append(int.from_bytes(node[suffix][:16], "little") + 2**128)
                else:
                    ys.append(int.from_bytes(node[suffix][16:], "little"))
            fs.append([(int.from_bytes(node[suffix_offset + i][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128) if suffix_offset + i in node else 0
                                                for i in range(128)
                                                for j in range(2)])

    D_serialized, ipa_proof = make_ipa_multiproof(Cs, fs, zs, ys, display_times)

    # All commitments, but without any duplications. These are for sending over the wire as part of the proof
    nodes_sorted_by_path = sorted(nodes_by_path.items())
    commitments_sorted_by_path_serialized = []
    for path, node_type_and_node in nodes_sorted_by_path[1:]:
        node_type, node = node_type_and_node
        if node_type in [VERKLE_PROOF_COMMITMENT_TYPE_INNER, VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION]:
            commitments_sorted_by_path_serialized.append(node["commitment"].serialize())
        elif node_type == VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1:
            commitments_sorted_by_path_serialized.append(node["C1"].serialize())
        elif node_type == VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C2:
            commitments_sorted_by_path_serialized.append(node["C2"].serialize())

    stems_with_extension = set(stem for stem in extension_present_by_stem if extension_present_by_stem[stem])
    other_stems = sorted(list(stem for stem in other_stems if stem not in stems_with_extension))

    log_time_if_eligible("   Serialized commitments", 30, display_times)

    return depths, extension_present, commitments_sorted_by_path_serialized, other_stems, D_serialized, ipa_proof


def check_verkle_proof(verkle_root, keys, values, proof, display_times=True):
    """
    Checks verkle trie proof according to
    https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html
    """

    start_logging_time_if_eligible("   Starting proof check", display_times)

    # Unpack the proof
    depths, extension_present, commitments_sorted_by_path_serialized, other_stems, D_serialized, ipa_proof = proof
    commitments_sorted_by_path = [Point().deserialize(verkle_root)] + [Point().deserialize(x) for x in commitments_sorted_by_path_serialized]

    # Find all stems
    stems = sorted(list(set([get_stem(key) for key in keys])))
    assert other_stems == sorted(other_stems)
    depths_by_stem = {}
    extension_present_by_stem = {}
    stems_with_extension = set()
    other_stems_used = set()

    # The commitments in the proof are sorted by path in the verkle trie. So we first need to recreate a list of all paths
    # that are required for the proof, so that then we can associate all commitments with their path.
    commitment_types_by_path = {}
    all_paths_and_zs = set()

    leaf_values_by_path_and_z = {}
    other_stems_by_prefix = {}

    for stem, depth, extpres in zip(stems, depths, extension_present):
        depths_by_stem[stem] = depth
        extension_present_by_stem[stem] = extpres
        if extpres == VERKLE_PROOF_EXTENSION_PRESENT_PRESENT:
            stems_with_extension.add(stem)
    
    # Find all required indices
    for key, value in zip(keys, values):
        stem = get_stem(key)
        depth = depths_by_stem[stem]
        extpres = extension_present_by_stem[stem]
        for i in range(depth):
            commitment_types_by_path[stem[:i]] = VERKLE_PROOF_COMMITMENT_TYPE_INNER
            all_paths_and_zs.add((stem[:i], stem[i]))

        if extpres in [VERKLE_PROOF_EXTENSION_PRESENT_OTHERSTEM, VERKLE_PROOF_EXTENSION_PRESENT_PRESENT]:

            commitment_types_by_path[stem[:depth]] = VERKLE_PROOF_COMMITMENT_TYPE_EXTENSION
            all_paths_and_zs.add((stem[:depth], 0))
            all_paths_and_zs.add((stem[:depth], 1))

            leaf_values_by_path_and_z[(stem[:depth], 0)] = 1

            if extpres == VERKLE_PROOF_EXTENSION_PRESENT_PRESENT:
                suffix = get_suffix(key)

                all_paths_and_zs.add((stem[:depth], 2 + (suffix // 128)))

                leaf_values_by_path_and_z[(stem[:depth], 1)] = int.from_bytes(stem, "little")

                suffix_tree_path = stem[:depth] + bytes([2 if suffix < 128 else 3])
                commitment_types_by_path[suffix_tree_path] = VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C1 if suffix < 128 \
                                                                else VERKLE_PROOF_COMMITMENT_TYPE_SUFFIX_TREE_C2
                all_paths_and_zs.add((suffix_tree_path, 2 * suffix % 256))
                all_paths_and_zs.add((suffix_tree_path, (2 * suffix + 1) % 256))

                value_lower = int.from_bytes(value[:16], "little") + 2**128 if value != None else 0
                value_upper = int.from_bytes(value[16:], "little") if value != None else 0

                leaf_values_by_path_and_z[(suffix_tree_path, 2 * suffix % 256)] = value_lower
                leaf_values_by_path_and_z[(suffix_tree_path, (2 * suffix + 1) % 256)] = value_upper
            elif extpres == VERKLE_PROOF_EXTENSION_PRESENT_OTHERSTEM:
                # The proof indicates that an extension node for a different stem was found in the tree
                # We need to verify this is the case by looking up the other stem in "other_stems"
                # and verifying that the extension node at the site is indeed for the other stem

                # First check if the extension node is already included in the proof because the other stem
                # is already part of this proof. In this special case, we don't have to do anything because the 
                # extension proof for the other stem will already do all the work.
                other_stem = None

                # The stem not having an extension node means that the value has never been set:
                if value is not None:
                    return False

                # First check if the other stem is already in one of the revealed values
                # TODO: Convert this to binary search to prevent DOS vectors
                for o in stems_with_extension:
                    if o[:depth] == stem[:depth]:
                        assert other_stem is None
                        other_stem = o
                
                if other_stem is None:
                    # TODO: `other_stem` search is linear search which should work perfectly in average case but
                    # is a DOS vector. Need to employ binary search
                    for o in other_stems:
                        if o[:depth] == stem[:depth]:
                            assert other_stem is None
                            other_stem = o

                    assert other_stem is not None
                    other_stems_used.add(other_stem)

                    # Now we need to add this extension node to the proof to show that our original stem wasn't
                    # present
                    leaf_values_by_path_and_z[(other_stem[:depth], 1)] = int.from_bytes(other_stem, "little")

                other_stems_by_prefix[stem[:depth]] = other_stem

        elif extpres == VERKLE_PROOF_EXTENSION_PRESENT_NOEXTENSION:
            # Prover can only claim extension is not present if value was never written
            if value is not None:
                return False

            leaf_values_by_path_and_z[(stem[:depth - 1], stem[depth - 1])] = 0
        else:
            # Invalid value for extpres
            return False, None

    # In order to assure uniqueness of the proof, we want the set of other stems to be exact and not include any extras
    # that weren't necessary.
    assert set(other_stems) == other_stems_used

    all_paths = sorted(commitment_types_by_path.keys())
    assert len(all_paths) == len(commitments_sorted_by_path)
    all_paths_and_zs = sorted(all_paths_and_zs)

    log_time_if_eligible("   Computed indices", 30, display_times)

    # Step 0: recreate the commitment list sorted by indices
    commitments_by_path = {path: commitment for path, commitment in zip(all_paths, commitments_sorted_by_path)}
    commitments_by_path_and_z = {(path, z): commitments_by_path[path] for path, z in all_paths_and_zs}
    
    ys_by_path_and_z = {}
    for path, z in all_paths_and_zs:
        child_path = path + bytes([z])
        ys_by_path_and_z[(path, z)] = leaf_values_by_path_and_z[(path, z)] if (path, z) in leaf_values_by_path_and_z \
                                      else int.from_bytes(commitments_by_path[child_path].serialize(), "little") % MODULUS

    Cs = list(map(lambda x: x[1], sorted(commitments_by_path_and_z.items())))
    
    zs = list(map(lambda x: x[1], sorted(all_paths_and_zs)))
    
    ys = list(map(lambda x: x[1], sorted(ys_by_path_and_z.items())))

    log_time_if_eligible("   Recreated commitment lists", 30, display_times)

    update_hint = depths_by_stem, extension_present_by_stem, commitments_by_path, other_stems_by_prefix

    return check_ipa_multiproof(Cs, zs, ys, [D_serialized, ipa_proof], display_times), update_hint


def compute_updated_verkle_root(verkle_root, keys, values, updated_values, update_hint, display_times=True):
    """
    Computes the updated verkle root

    `updated_values` contains new updated values in the same order as the keys/values.
    Can be `None` for any value that does not need updating.
    Returns the updated verkle root
    """

    depths_by_stem, extension_present_by_stem, commitments_by_path, other_stems_by_prefix = update_hint

    updated_stems = {}

    for key, old_value, updated_value in zip(keys, values, updated_values):
        if updated_value is not None:
            stem = get_stem(key)
            suffix = get_suffix(key)
            if stem not in updated_stems:
                updated_stems[stem] = {}
            updated_stems[stem][suffix] = (old_value, updated_value)

    commitment_updates_by_path = {}

    # Collect all updates to suffix tree nodes

    updated_stems_by_prefix = {}
    for stem, update in updated_stems.items():
        depth = depths_by_stem[stem]
        prefix = stem[:depth]
        if prefix in updated_stems_by_prefix:
            updated_stems_by_prefix[prefix].add(stem)
        else:
            updated_stems_by_prefix[prefix] = set([stem])
        if extension_present_by_stem[stem] == VERKLE_PROOF_EXTENSION_PRESENT_PRESENT:
            extension_path = stem[:depth]

            c1_field_update = 0
            if any(i in update for i in range(128)):
                c1_update_vals = {2 * i + j: (MODULUS + int.from_bytes(update[i][1][16 * j: 16 * (j + 1)], "little") + (1 - j) * 2**128
                                              - (int.from_bytes(update[i][0][16 * j: 16 * (j + 1)], "little") + (1 - j) * 2**128 if update[i][0] is not None else 0)) % MODULUS
                                    for i in range(128)
                                    for j in range(2)
                                    if i in update}
                c1_update = ipa_utils.pedersen_commit_sparse(c1_update_vals)
                c1_path = extension_path + bytes([2])
                old_commitment = commitments_by_path[c1_path]
                new_commitment = old_commitment.dup().add(c1_update)
                c1_field_update = (MODULUS + commitment_to_field(new_commitment) - commitment_to_field(old_commitment)) % MODULUS

            c2_field_update = 0
            if any(i in update for i in range(128, 256)):
                c2_update_vals = {2 * i + j: (MODULUS + int.from_bytes(update[128 + i][1][16 * j: 16 * (j + 1)], "little") + (1 - j) * 2**128
                                              - (int.from_bytes(update[128 + i][0][16 * j: 16 * (j + 1)], "little") + (1 - j) * 2**128 if update[128 + i][0] is not None else 0)) % MODULUS
                                    for i in range(128)
                                    for j in range(2)
                                    if 128 + i in update}
                c2_update = ipa_utils.pedersen_commit_sparse(c2_update_vals)
                c2_path = extension_path + bytes([3])
                old_commitment = commitments_by_path[c2_path]
                new_commitment = old_commitment.dup().add(c2_update)
                c2_field_update = (MODULUS + commitment_to_field(new_commitment) - commitment_to_field(old_commitment)) % MODULUS

            commitment_update = ipa_utils.pedersen_commit_sparse({2: c1_field_update, 3: c2_field_update})
            old_commitment = commitments_by_path[extension_path]
            new_commitment = old_commitment.dup().add(commitment_update)
            update["commitment"] = new_commitment
            update["commitment_field"] = commitment_to_field(new_commitment)
        
        else:
            if extension_present_by_stem[stem] == VERKLE_PROOF_EXTENSION_PRESENT_OTHERSTEM:
                updated_stems_by_prefix[prefix].add(other_stems_by_prefix[prefix])

            c1_vals = {2 * i + j: int.from_bytes(update[i][1][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                    for i in range(128)
                                                    for j in range(2)
                                                    if i in update}
            C1 = ipa_utils.pedersen_commit_sparse(c1_vals)

            c2_vals = {2 * i + j: int.from_bytes(update[128 + i][1][16 * j:16 * (j + 1)], "little") + (1 - j) * 2**128
                                                    for i in range(128)
                                                    for j in range(2)
                                                    if 128 + i in update}
            C2 = ipa_utils.pedersen_commit_sparse(c2_vals)

            C1_field = commitment_to_field(C1)
            C2_field = commitment_to_field(C2)

            commitment = ipa_utils.pedersen_commit_sparse({0: 1, 
                                                           1: int.from_bytes(stem, "little"),
                                                           2: C1_field, 
                                                           3: C2_field})
            update["commitment"] = commitment
            update["commitment_field"] = commitment_to_field(commitment)

    # Now we need to merge all the updated extension nodes back into the trie nodes.
    # First we insert all the stem updates into an update tree, which we can then process to get the
    # root update
    update_tree = {}

    def insert_into_update_tree(prefix, field_update):
        """
        Helper function to insert field updates into the update tree
        """
        current_node = update_tree
        for index in prefix[:-1]:
            if index not in current_node:
                current_node[index] = {}
            current_node = current_node[index]
        assert prefix[-1] not in current_node
        current_node[prefix[-1]] = field_update

    for prefix, stems in updated_stems_by_prefix.items():
        if prefix in commitments_by_path:
            old_commitment = commitments_by_path[prefix]
            old_field = commitment_to_field(old_commitment)
        else:
            old_field = 0
        
        if len(stems) == 1:
            stem = list(stems)[0]
            new_field = updated_stems[stem]["commitment_field"]
        else:
            assert len(stems) > 1
            # We need to build the subtree from here
            # We can simply "abuse" the function `verkle_add_missing_commitments`, despite the nodes being only partial
            # tree nodes
            tree_at_prefix = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
            start_depth = len(prefix)
            for stem in stems:
                current_node = tree_at_prefix
                depth = start_depth

                if stem in updated_stems:
                    commitment, commitment_field = updated_stems[stem]["commitment"], updated_stems[stem]["commitment_field"]
                else:
                    # This is the stem that was originally at this prefix path -- it wasn't updated but we do need
                    # to insert it into the tree
                    commitment = commitments_by_path[prefix]
                    commitment_field = commitment_to_field(commitment)
                new_suffix_tree = {"node_type": VERKLE_TRIE_NODE_TYPE_SUFFIX_TREE, "commitment": commitment, "commitment_field": commitment_field, "stem": stem}

                while current_node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
                    previous_node = current_node
                    index = stem[depth]
                    depth += 1
                    if index in current_node:
                        current_node = current_node[index]
                    else:
                        break

                if current_node["node_type"] == VERKLE_TRIE_NODE_TYPE_INNER:
                    current_node[index] = new_suffix_tree
                else:
                    old_suffix_tree = current_node
                    old_stem = old_suffix_tree["stem"]

                    new_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
                    previous_node[index] = new_inner_node
                    previous_node = new_inner_node
                    
                    while old_stem[depth] == stem[depth]:
                        index = stem[depth]
                        new_inner_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}
                        previous_node[index] = new_inner_node
                        previous_node = new_inner_node
                        depth += 1

                    new_inner_node[stem[depth]] = new_suffix_tree
                    new_inner_node[old_stem[depth]] = old_suffix_tree

            verkle_add_missing_commitments(tree_at_prefix)
            new_field = tree_at_prefix["commitment_field"]

        field_update = (MODULUS + new_field - old_field) % MODULUS

        insert_into_update_tree(prefix, field_update)

    def compute_update(path, node):
        """
        Helper function to compute the updates in the update tree
        """
        field_updates = {}
        for index, update_node in node.items():
            if isinstance(update_node, dict):
                field_updates[index] = compute_update(path + bytes([index]), update_node)
            else:
                field_updates[index] = update_node
        commitment_update = ipa_utils.pedersen_commit_sparse(field_updates)
        old_commitment = commitments_by_path[path]
        new_commitment = old_commitment.dup().add(commitment_update)
        if path == b'':
            return new_commitment
        field_update = (MODULUS + commitment_to_field(new_commitment) - commitment_to_field(old_commitment)) % MODULUS
        return field_update

    root_update = compute_update(b'', update_tree)
    
    log_time_if_eligible("   Computed updated verkle root", 30, display_times)

    return root_update


if __name__ == "__main__":

    BASIS = generate_basis(WIDTH)
    ipa_utils = IPAUtils(BASIS["G"], BASIS["Q"], primefield)

    # Build a random verkle trie
    root_node = {"node_type": VERKLE_TRIE_NODE_TYPE_INNER}

    values = {}

    for i in range(NUMBER_STEMS):
        stem = randint(0, 2**248-1).to_bytes(31, "little")
        for i in range(CHUNKS_PER_STEM):
            key = stem + bytes([randint(0, 2**8-1)])
            value = randint(0, 2**256-1).to_bytes(32, "little")
            update_verkle_tree_nocommitmentupdate(root_node, key, value)
            values[key] = value
    
    average_depth = get_average_depth(root_node)
        
    print("Inserted {0} elements for an average depth of {1:.3f}".format(NUMBER_CHUNKS, average_depth), file=sys.stderr)
    print("Average depth = {0:.3f} without counting suffix trees (stem tree only)".format(average_depth - 2), file=sys.stderr)

    time_a = time()
    verkle_add_missing_commitments(root_node)
    time_b = time()

    print("Computed verkle root in {0:.3f} s".format(time_b - time_a), file=sys.stderr)

    time_a = time()
    assert values == check_valid_tree(root_node)
    time_b = time()
    
    print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)

    if NUMBER_ADDED_STEMS > 0:
        time_x = time()
        for i in range(NUMBER_ADDED_STEMS):
            key = randint(0, 2**256-1).to_bytes(32, "little")
            value = randint(0, 2**256-1).to_bytes(32, "little")
            update_verkle_tree(root_node, key, value)
            values[key] = value
        time_y = time()
            
        print("Additionally inserted {0} stems in {1:.3f} s".format(NUMBER_ADDED_STEMS, time_y - time_x), file=sys.stderr)
        print("Keys in tree now: {0}, average depth: {1:.3f}".format(get_total_depth(root_node)[1], get_average_depth(root_node)), file=sys.stderr)

        time_a = time()
        assert values == check_valid_tree(root_node)
        time_b = time()
        
        print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)

    if NUMBER_ADDED_CHUNKS > 0:

        time_x = time()
        existing_keys = list(values.keys())
        for i in range(NUMBER_ADDED_CHUNKS):
            stem = get_stem(choice(existing_keys))
            suffix = randint(0, 255)
            key = stem + bytes([suffix])
            value = randint(0, 2**256-1).to_bytes(32, "little")
            update_verkle_tree(root_node, key, value)
            values[key] = value
        time_y = time()
            
        print("Additionally inserted {0} chunks in {1:.3f} s".format(NUMBER_ADDED_CHUNKS, time_y - time_x), file=sys.stderr)
        print("Keys in tree now: {0}, average depth: {1:.3f}".format(get_total_depth(root_node)[1], get_average_depth(root_node)), file=sys.stderr)

        time_a = time()
        assert values == check_valid_tree(root_node)
        time_b = time()
        
        print("[Checked tree valid: {0:.3f} s]".format(time_b - time_a), file=sys.stderr)

    all_keys = list(values.keys())
    shuffle(all_keys)

    keys_in_proof = all_keys[:NUMBER_EXISTING_KEYS_PROOF]
    values_in_proof = [values[key] for key in keys_in_proof]

    for i in range(NUMBER_RANDOM_STEMS_PROOF):
        key = randint(0, 2**256-1).to_bytes(32, "little")
        keys_in_proof.append(key)
        values_in_proof.append(values[key] if key in values else None)

    for i in range(NUMBER_RANDOM_CHUNKS_PROOF):
        stem = get_stem(choice(existing_keys))
        suffix = randint(0, 255)
        key = stem + bytes([suffix])        
        keys_in_proof.append(key)
        values_in_proof.append(values[key] if key in values else None)

    time_a = time()
    proof = make_verkle_proof(root_node, keys_in_proof)
    time_b = time()
    
    proof_size = get_proof_size(proof)
    proof_time = time_b - time_a
    
    print("Computed proof for {0} keys (size = {1} bytes) in {2:.3f} s".format(len(keys_in_proof), proof_size, time_b - time_a), file=sys.stderr)
    print("Witness size per key: {0:.3f} bytes".format(proof_size / len(keys_in_proof)))

    updated_values = [None] * len(values_in_proof)
    for i in range(NUMBER_VALUES_UPDATED):
        j = randint(0, len(updated_values) - 1)
        while updated_values[j] is not None:
            j = randint(0, len(updated_values) - 1)
        updated_values[j] = randint(0, 2**256-1).to_bytes(32, "little")

    time_a = time()
    r, update_hint =  check_verkle_proof(root_node["commitment"].serialize(), keys_in_proof, values_in_proof, proof)
    assert r
    time_b = time()
    check_time = time_b - time_a

    print("Checked proof in {0:.3f} s".format(time_b - time_a), file=sys.stderr)

    time_a = time()
    updated_root = compute_updated_verkle_root(root_node["commitment"].serialize(), keys_in_proof, values_in_proof, updated_values, update_hint)
    time_b = time()
    check_time = time_b - time_a

    print("Computed root update in {0:.3f} s".format(time_b - time_a), file=sys.stderr)

    time_a = time()
    for key, updated_value in zip(keys_in_proof, updated_values):
        if updated_value is not None:
            update_verkle_tree(root_node, key, updated_value)
    time_b = time()
    check_time = time_b - time_a

    assert root_node["commitment"] == updated_root

    print("Verified updated root in {0:.3f} s".format(time_b - time_a), file=sys.stderr)
