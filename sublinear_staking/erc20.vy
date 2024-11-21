# Minimal ERC-20 contract

balances: HashMap[address, uint256]
allowances: HashMap[address, HashMap[address, uint256]]
total_supply: uint256

@external
def balanceOf(_owner: address) -> uint256:
    return self.balances[_owner]

@external
def transfer(_to: address, _value: uint256) -> bool:
    assert self.balances[msg.sender] >= _value, "Insufficient balance"
    self.balances[msg.sender] -= _value
    self.balances[_to] += _value
    return True

@external
def transferFrom(_from: address, _to: address, _value: uint256) -> bool:
    assert self.balances[_from] >= _value, "Insufficient balance"
    assert self.allowances[_from][msg.sender] >= _value, "Insufficient allowance"
    self.allowances[_from][msg.sender] -= _value
    self.balances[_from] -= _value
    self.balances[_to] += _value
    return True

@external
def approve(_spender: address, _value: uint256) -> bool:
    self.allowances[msg.sender][_spender] = _value
    return True

@external
def mint(_to: address, _value: uint256):
    self.balances[_to] += _value
    self.total_supply += _value

@external
def totalSupply() -> uint256:
    return self.total_supply

