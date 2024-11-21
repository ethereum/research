#pragma version >0.3.10

# Basic implementation of a sublinear staking contract. Stake coins, and you get
# return proportional to coins staked ** 0.75. Returns last as long as coins last

stakedAmount: public(HashMap[address, uint256])
stakeLastUpdated: public(HashMap[address, uint256])
stakedTokenAddress: address
uniqueidTokenAddress: address
uniqueidTokenCollection: uint256
totalPayoutPerSlot: uint256
liabilities: uint256
liabilitiesLastUpdated: uint256

@view
def intSqrt(input: uint256) -> uint256:
    o: uint256 = input
    if o == 0:
        return 0
    for i:uint256 in range(32):
        o = (o + input // o) // 2
    return o
    
@view
def getReturnPerSlot(x: uint256) -> uint256:
    sqrtX: uint256 = self.intSqrt(x)
    return sqrtX * self.intSqrt(sqrtX)
    
# Define the interface for the ERC-1155 contract
interface ERC1155:
    def balanceOf(_owner: address, _id: uint256) -> uint256: view
    
# ERC-20 Interface in Vyper
interface ERC20:
    def transfer(_to: address, _value: uint256) -> bool: nonpayable
    def transferFrom(_from: address, _to: address, _value: uint256) -> bool: nonpayable
    def balanceOf(_owner: address) -> uint256: view

@view
def isEligible(user: address) -> bool:
# Create an instance of the ERC-1155 contract
    c: ERC1155 = ERC1155(self.uniqueidTokenAddress)
    
    # Get the balance of the user for the specified token ID
    balance: uint256 = staticcall c.balanceOf(user, self.uniqueidTokenCollection)
    
    # Return True if balance is greater than zero, else False
    return balance > 0

# Setup global variables
@deploy
def __init__(stakedTokenAddress: address,
             uniqueidTokenAddress: address,
             uniqueidTokenCollection: uint256):
    self.stakedTokenAddress = stakedTokenAddress
    self.uniqueidTokenAddress = uniqueidTokenAddress
    self.uniqueidTokenCollection = uniqueidTokenCollection
    
@external
def stake(amount: uint256):
    assert self.isEligible(msg.sender)
    token: ERC20 = ERC20(self.stakedTokenAddress)
    if self.stakedAmount[msg.sender] > 0:
        self._unstake()
    returnPerSlot: uint256 = self.getReturnPerSlot(amount)
    self.stakedAmount[msg.sender] = amount
    self.stakeLastUpdated[msg.sender] = block.timestamp
    self.liabilities += (block.timestamp - self.liabilitiesLastUpdated) * self.totalPayoutPerSlot
    self.liabilities += amount
    self.liabilitiesLastUpdated = block.timestamp
    self.totalPayoutPerSlot += returnPerSlot
    success: bool = extcall token.transferFrom(msg.sender, self, amount)
    assert success
    
def _unstake():
    token: ERC20 = ERC20(self.stakedTokenAddress)
    returnPerSlot: uint256 = self.getReturnPerSlot(self.stakedAmount[msg.sender])
    deadline: uint256 = self.liabilitiesLastUpdated + (staticcall token.balanceOf(self) - self.liabilities) // max(self.totalPayoutPerSlot, 1)
    correctedNow: uint256 = min(block.timestamp, deadline)
    timeElapsed: uint256 = correctedNow - self.stakeLastUpdated[msg.sender]
    totalOut: uint256 = self.stakedAmount[msg.sender] + timeElapsed * returnPerSlot
    self.stakedAmount[msg.sender] = 0
    self.liabilities += (correctedNow - self.liabilitiesLastUpdated) * self.totalPayoutPerSlot
    self.liabilitiesLastUpdated = correctedNow
    self.totalPayoutPerSlot -= returnPerSlot
    self.liabilities -= totalOut
    extcall token.transfer(msg.sender, totalOut)
    
@external
def unstake():
    self._unstake()

@external
@view
def deadline() -> uint256:
    token: ERC20 = ERC20(self.stakedTokenAddress)
    return (
        self.liabilitiesLastUpdated +
        (staticcall token.balanceOf(self) - self.liabilities) //
        max(self.totalPayoutPerSlot, 1)
    )
