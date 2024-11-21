#pragma version >0.3.10

# Basic implementation of a sublinear staking contract. Stake coins, and you get
# return proportional to coins staked ** 0.75. Returns last as long as coins last

from ethereum.ercs import IERC20 as ERC20

# Define the interface for the ERC-1155 contract
interface ERC1155:
    def balanceOf(_owner: address, _id: uint256) -> uint256: view

stakedAmount: public(HashMap[address, uint256])
stakeLastUpdated: public(HashMap[address, uint256])

STAKED_TOKEN_ADDRESS: immutable(ERC20)
UNIQUEID_TOKEN_ADDRESS: immutable(ERC1155)
UNIQUEID_TOKEN_COLLECTION: immutable(uint256)

totalPayoutPerSlot: uint256
liabilities: uint256
liabilitiesLastUpdated: uint256

# If you stake x coins, this is the return you get per slot
@view
def getReturnPerSlot(x: uint256) -> uint256:
    sqrtX: uint256 = isqrt(x)
    return sqrtX * isqrt(sqrtX)
    
    
@view
def isEligible(user: address) -> bool:
    # Get the balance of the user for the specified token ID
    balance: uint256 = staticcall UNIQUEID_TOKEN_ADDRESS.balanceOf(
        user,
        UNIQUEID_TOKEN_COLLECTION
    )
    # Return True if balance is greater than zero, else False
    return balance > 0

# Setup global variables
@deploy
def __init__(stakedTokenAddress: address,
             uniqueidTokenAddress: address,
             uniqueidTokenCollection: uint256):
    STAKED_TOKEN_ADDRESS = ERC20(stakedTokenAddress)
    UNIQUEID_TOKEN_ADDRESS = ERC1155(uniqueidTokenAddress)
    UNIQUEID_TOKEN_COLLECTION = uniqueidTokenCollection

# Stake the specified number of tokens
@external
def stake(amount: uint256):
    assert self.isEligible(msg.sender)
    assert self.stakedAmount[msg.sender] == 0
    returnPerSlot: uint256 = self.getReturnPerSlot(amount)
    self.stakedAmount[msg.sender] = amount
    self.stakeLastUpdated[msg.sender] = block.timestamp
    # The contract tracks liabilities and totalPayoutPerSlot, so that it knows
    # how long it can keep paying rewards
    self.liabilities += (
        (block.timestamp - self.liabilitiesLastUpdated)
        * self.totalPayoutPerSlot
    )
    self.liabilities += amount
    self.liabilitiesLastUpdated = block.timestamp
    self.totalPayoutPerSlot += returnPerSlot
    success: bool = extcall STAKED_TOKEN_ADDRESS.transferFrom(
        msg.sender,
        self,
        amount,
        default_return_value=True
    )
    assert success

# Remove your stake, plus any returns
def _unstake() -> uint256:
    returnPerSlot: uint256 = self.getReturnPerSlot(self.stakedAmount[msg.sender])
    correctedNow: uint256 = min(block.timestamp, self._deadline())
    timeElapsed: uint256 = correctedNow - self.stakeLastUpdated[msg.sender]
    totalOut: uint256 = self.stakedAmount[msg.sender] + timeElapsed * returnPerSlot
    self.stakedAmount[msg.sender] = 0
    self.liabilities += (
        (correctedNow - self.liabilitiesLastUpdated)
        * self.totalPayoutPerSlot
    )
    self.liabilitiesLastUpdated = correctedNow
    self.totalPayoutPerSlot -= returnPerSlot
    self.liabilities -= totalOut
    success: bool = extcall STAKED_TOKEN_ADDRESS.transfer(
        msg.sender,
        totalOut,
        default_return_value=True
    )
    assert success
    return totalOut
    
@external
def unstake() -> uint256:
    return self._unstake()

# How long the contract can keep paying returns
@view
def _deadline() -> uint256:
    return (
        self.liabilitiesLastUpdated
        + (staticcall STAKED_TOKEN_ADDRESS.balanceOf(self) - self.liabilities)
        // max(self.totalPayoutPerSlot, 1)
    )

@external
@view
def deadline() -> uint256:
    return self._deadline()
