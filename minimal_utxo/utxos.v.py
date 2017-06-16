utxos: public({owner: address, value: num}[bytes32])

def __init__():
    self.utxos[as_bytes32(1)] = {owner: msg.sender, value: 4294967296}

def tx(in1: bytes32, in2: bytes32, out1: address, value1: num,
    out2: address, value2: num, v: num256, r: num256, s: num256) -> bytes32:
    sighash = sha3(concat(in1, in2, as_bytes32(out1), as_bytes32(value1), as_bytes32(out2), as_bytes32(value2)))
    sender = ecrecover(sighash, v, r, s)
    assert self.utxos[in1].owner == sender or not in1
    assert self.utxos[in2].owner == sender or not in2
    value = self.utxos[in1].value + self.utxos[in2].value
    assert value == value1 + value2
    self.utxos[in1] = {owner: None, value: None}
    self.utxos[in2] = {owner: None, value: None}
    self.utxos[sighash] = {owner: out1, value: value1}
    self.utxos[as_bytes32(bitwise_xor(as_num256(sighash), as_num256(1)))] = \
        {owner: out2, value: value2}
    return sighash
