# Minimal ERC-1155 contract

balances: HashMap[address, HashMap[uint256, uint256]]

@external
def balanceOf(_owner: address, _id: uint256) -> uint256:
    return self.balances[_owner][_id]

@external
def mint(_to: address, _id: uint256, _amount: uint256):
    self.balances[_to][_id] += _amount
