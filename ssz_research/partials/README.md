### What is an SSZ Partial?

An SSZ partial is an object that can stand in for an SSZ object in any function, but which only contains some of the elements in the SSZ object. It also contains Merkle proofs that prove that the values included in the SSZ partial actually are the values from the original SSZ object; this can be verified by computing the Merkle root of the SSZ partial and verifying that it matches the root of the original SSZ object.

See `test_ssz_partial.py` for more details.
