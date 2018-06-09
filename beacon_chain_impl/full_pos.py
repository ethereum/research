from hashlib import blake2s
blake = lambda x: blake2s(x).digest()
from ethereum.utils import normalize_address, hash32, trie_root, \
    big_endian_int, address, int256, encode_hex, decode_hex, encode_int, \
    big_endian_to_int
from rlp.sedes import big_endian_int, Binary, binary, CountableList
import rlp
import bls
import random

class BeaconBlock(rlp.Serializable):
    
    fields = [
        ('parent_hash', hash32),
        ('skip_count', int256),
        ('randao_reveal', hash32),
        ('attestation_bitmask', binary),
        ('attestation_aggregate_sig', int256),
        ('ffg_signer_list', binary),
        ('ffg_aggregate_sig', int256),
        ('main_chain_ref', hash32),
        ('state_hash', hash32),
        ('height', int256),
        ('sig', int256)
    ]

    def __init__(self,
                 parent_hash=b'\x00'*32, skip_count=0, randao_reveal=b'\x00'*32,
                 attestation_bitmask=b'', attestation_aggregate_sig=0,
                 ffg_signer_list=b'', ffg_aggregate_sig=0, main_chain_ref=b'\x00'*32,
                 state_hash=b'\x00'*32, height=0, sig=0):
        # at the beginning of a method, locals() is a dict of all arguments
        fields = {k: v for k, v in locals().items() if k != 'self'}
        super(BlockHeader, self).__init__(**fields)

def quick_sample(seed, validator_count, sample_count):
    k = 0
    while 256**k < n:
        k += 1
    o = []; source = seed; pos = 0
    while len(o) < sample_count:
        if pos + k > 32:
            source = blake(source)
            pos = 0
        m = big_endian_to_int(source[pos:pos+k])
        if n * (m // n + 1) <= 256**k:
            o.append(m % n)
        pos += k
    return o

privkeys = [int.from_bytes(blake2s(str(i).encode('utf-8'))) for i in range(3000)]

def mock_make_child(parent_state, skips, ):
    attest

    fields = [
        ('parent_hash', hash32),
        ('skip_count', int256),
        ('randao_reveal', hash32),
        ('attestation_bitmask', binary),
        ('attestation_aggregate_sig', int256),
        ('ffg_signer_list', binary),
        ('ffg_aggregate_sig', int256),
        ('main_chain_ref', hash32),
        ('state_hash', hash32),
        ('height', int256),
        ('sig', int256)
    ]
