from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import EthereumTester
from vyper import compile_code

# Set up Web3 and Ethereum Tester
eth_tester = EthereumTester()
w3 = Web3(EthereumTesterProvider(eth_tester))

# Accounts
accounts = w3.eth.accounts
a1, a2, a3, a4, a5 = accounts[0], accounts[1], accounts[2], accounts[3], accounts[4]

# Vyper code for the main staking contract
with open('code.vy') as f:
    staking_source_code = f.read()
with open('erc20.vy') as f:
    erc20_vyper_code = f.read()
with open('erc1155.vy') as f:
    erc1155_vyper_code = f.read()

# Compile the staking contract
compiled_staking_contract = compile_code(staking_source_code, output_formats=['abi', 'bytecode'])
staking_contract_interface = {
    'abi': compiled_staking_contract['abi'],
    'bytecode': compiled_staking_contract['bytecode']
}

# Compile the ERC-1155 contract
compiled_erc1155 = compile_code(erc1155_vyper_code, output_formats=['abi', 'bytecode'])
erc1155_interface = {
    'abi': compiled_erc1155['abi'],
    'bytecode': compiled_erc1155['bytecode']
}

# Compile the ERC-20 contract
compiled_erc20 = compile_code(erc20_vyper_code, output_formats=['abi', 'bytecode'])
erc20_interface = {
    'abi': compiled_erc20['abi'],
    'bytecode': compiled_erc20['bytecode']
}

# Helper function to deploy contracts
def deploy_contract(w3, abi, bytecode, constructor_args=(), deployer=a1):
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = Contract.constructor(*constructor_args).transact({'from': deployer})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return w3.eth.contract(address=tx_receipt.contractAddress, abi=abi)

# Step 3: Deploy ERC-1155 contract and mint tokens
erc1155_contract = deploy_contract(w3, erc1155_interface['abi'], erc1155_interface['bytecode'])
A = erc1155_contract.address
cID = 1  # Collection ID

# Mint 1 unit of token ID cID to a1, a2, a3, a4
for account in [a1, a2, a3, a4]:
    tx_hash = erc1155_contract.functions.mint(account, cID, 1).transact({'from': a1})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 4: Deploy ERC-20 contract and mint tokens
erc20_contract = deploy_contract(w3, erc20_interface['abi'], erc20_interface['bytecode'])
T = erc20_contract.address

# Mint 10**18 units to a1, a2, a3, a4, a5
initial_balance = 10**18
for account in [a1, a2, a3, a4, a5]:
    tx_hash = erc20_contract.functions.mint(account, initial_balance).transact({'from': a1})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 5: Deploy the staking contract (C) with T, A, cID
staking_contract = deploy_contract(
    w3,
    staking_contract_interface['abi'],
    staking_contract_interface['bytecode'],
    constructor_args=(T, A, cID, 1)
)
C = staking_contract.address

# Mint 10**18 units to the staking contract C
tx_hash = erc20_contract.functions.mint(C, 10**17).transact({'from': a1})
w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 6: Approve the staking contract to spend tokens for each account
for account in [a1, a2, a3, a4, a5]:
    tx_hash = erc20_contract.functions.approve(C, initial_balance).transact({'from': account})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 7: Each account attempts to stake tokens
stake_amounts = {
    a1: 1,
    a2: 10**9,
    a3: 10**18,
    a4: 10**18,
    a5: 10**18,
}

print("\nStaking attempts:")
for account in [a1, a2, a3, a4, a5]:
    amount = stake_amounts[account]
    try:
        tx_hash = staking_contract.functions.stake(amount).transact({'from': account})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Stake successful for account {account} with amount {amount}")
        success = True
    except Exception as e:
        print(f"Stake failed for account {account} with amount {amount}: {e}")
        success = False
    assert success == (account != a5)

# Step 8: Fast forward 1000 blocks
eth_tester = w3.provider.ethereum_tester
now = w3.eth.get_block('latest')['timestamp']
fundedUntil = staking_contract.functions.fundedUntil().call()
assert 1000 < fundedUntil - now < 2000
print(f"Before fast forward: {now}")
eth_tester.mine_blocks(1000)
print(f"FundedUntil minus now: {fundedUntil - now}")
now2 = w3.eth.get_block('latest')['timestamp']
print(f"After fast forward: {now2}")

# Step 9: Each account except a4 unstakes their tokens and prints the amounts
print("\nUnstaking attempts (first round, fundedUntil not hit):")
for account in [a1, a2, a3]:
    tx_hash = staking_contract.functions.unstake().transact({'from': account})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    # Get the new balance
    balance = erc20_contract.functions.balanceOf(account).call()
    expected_return = int(stake_amounts[account] ** 0.75) * (now2 - now)
    actual_return = balance - 10**18
    assert 0.99 < actual_return / expected_return < 1.01
    print(f"Account {account} unstaked {amount}, new balance is {balance}")

# Step 10: fast forward more, and then a4 unstakes. But the contract
# runs out of money, so a4 does not get paid for the full 2000 blocks
fundedUntil = staking_contract.functions.fundedUntil().call()
print(f"FundedUntil minus now: {fundedUntil - now2}")
assert fundedUntil - now2 < 2000
eth_tester.mine_blocks(2000)
print("\nUnstaking attempts (second round, fundedUntil hit):")
for account in [a4]:
    tx_hash = staking_contract.functions.unstake().transact({'from': account})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    # Get the new balance
    balance = erc20_contract.functions.balanceOf(account).call()
    expected_return = int(stake_amounts[account] ** 0.75) * (fundedUntil - now)
    actual_return = balance - 10**18
    assert 0.99 < actual_return / expected_return < 1.01
    print(f"Account {account} unstaked {amount}, new balance is {balance}")
contract_balance = erc20_contract.functions.balanceOf(staking_contract.address).call()
print(f"Remaining balance: {contract_balance}")
