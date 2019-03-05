# Copy from Trinity
# https://github.com/ethereum/trinity/pull/341


"""Utilities for sparse binary merkle trees.
Merkle trees are represented as sequences of layers, from root to leaves. The root layer contains
only a single element, the leaves as many as there are data items in the tree. The data itself is
not considered to be part of the tree.
"""
from typing import (
    Iterable,
    Sequence,
    Union,
)

from cytoolz import (
    iterate,
    take,
)

from eth_typing import (
    Hash32,
)
from eth_utils import (
    ValidationError,
)

from hashlib import sha256
def hash_eth2(x): return sha256(x).digest()


from .merkle_normal import (  # noqa: F401
    _calc_parent_hash,
    _hash_layer,
    get_branch_indices,
    get_root,
    MerkleTree,
    MerkleProof,
)

from typing import (
    Tuple,
    TypeVar,
)

from eth_utils import (
    ValidationError,
)


VType = TypeVar('VType')


def update_tuple_item(tuple_data: Tuple[VType, ...],
                      index: int,
                      new_value: VType) -> Tuple[VType, ...]:
    """
    Update the ``index``th item of ``tuple_data`` to ``new_value``
    """
    list_data = list(tuple_data)

    try:
        list_data[index] = new_value
    except IndexError:
        raise ValidationError(
            "the length of the given tuple_data is {}, the given index {} is out of index".format(
                len(tuple_data),
                index,
            )
        )
    else:
        return tuple(list_data)


TreeHeight = 32
EmptyNodeHashes = tuple(
    take(TreeHeight, iterate(lambda node_hash: hash_eth2(node_hash + node_hash), b'\x00' * 32))
)


def get_merkle_proof(tree: MerkleTree, item_index: int) -> Iterable[Hash32]:
    """
    Read off the Merkle proof for an item from a Merkle tree.
    """
    if item_index < 0 or item_index >= len(tree[-1]) or tree[-1][item_index] == EmptyNodeHashes[0]:
        raise ValidationError("Item index out of range")

    branch_indices = get_branch_indices(item_index, len(tree))
    proof_indices = [i ^ 1 for i in branch_indices][:-1]  # get sibling by flipping rightmost bit
    return tuple(
        layer[proof_index]
        for layer, proof_index
        in zip(reversed(tree), proof_indices)
    )


def calc_merkle_tree_from_leaves(leaves: Sequence[Hash32]) -> MerkleTree:
    if len(leaves) == 0:
        raise ValueError("No leaves given")
    tree = tuple()  # type: ignore
    tree = (leaves,) + tree
    for i in range(TreeHeight):
        if len(tree[0]) % 2 == 1:
            tree = update_tuple_item(tree, 0, tree[0] + (EmptyNodeHashes[i],))
        tree = (_hash_layer(tree[0]),) + tree
    return tree


def get_merkle_root(leaves: Sequence[Hash32]) -> Hash32:
    """
    Return the Merkle root of the given 32-byte hashes.
    """
    return get_root(calc_merkle_tree_from_leaves(leaves))

