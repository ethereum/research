from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import EthereumTester
from vyper import compile_code

# Set up Web3 and Ethereum Tester
eth_tester = EthereumTester()
w3 = Web3(EthereumTesterProvider(eth_tester))

# Accounts
accounts = w3.eth.accounts
a1, a2, a3, a4 = accounts[0], accounts[1], accounts[2], accounts[3]

# Vyper code for the main staking contract
staking_source_code = open('code.vy').read()
erc20_vyper_code = open('erc20.vy').read()
erc1155_vyper_code = open('erc1155.vy').read()

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

# Mint 1 unit of token ID cID to a1, a2, a3
for account in [a1, a2, a3]:
    tx_hash = erc1155_contract.functions.mint(account, cID, 1).transact({'from': a1})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 4: Deploy ERC-20 contract and mint tokens
erc20_contract = deploy_contract(w3, erc20_interface['abi'], erc20_interface['bytecode'])
T = erc20_contract.address

# Mint 10**18 units to a1, a2, a3, a4
initial_balance = 10**18
for account in [a1, a2, a3, a4]:
    tx_hash = erc20_contract.functions.mint(account, initial_balance).transact({'from': a1})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 5: Deploy the staking contract (C) with T, A, cID
staking_contract = deploy_contract(
    w3,
    staking_contract_interface['abi'],
    staking_contract_interface['bytecode'],
    constructor_args=(T, A, cID)
)
C = staking_contract.address

# Mint 10**18 units to the staking contract C
tx_hash = erc20_contract.functions.mint(C, 3 * 10**16).transact({'from': a1})
w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 6: Approve the staking contract to spend tokens for each account
for account in [a1, a2, a3, a4]:
    tx_hash = erc20_contract.functions.approve(C, initial_balance).transact({'from': account})
    w3.eth.wait_for_transaction_receipt(tx_hash)

# Step 7: Each account attempts to stake tokens
stake_amounts = {
    a1: 1,
    a2: 10**9,
    a3: 10**18,
    a4: 10**18,
}

print("\nStaking attempts:")
for account in [a1, a2, a3, a4]:
    amount = stake_amounts[account]
    try:
        tx_hash = staking_contract.functions.stake(amount).transact({'from': account})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Stake successful for account {account} with amount {amount}")
    except Exception as e:
        print(f"Stake failed for account {account} with amount {amount}: {e}")

# Step 8: Fast forward 1000 blocks
eth_tester = w3.provider.ethereum_tester
now = w3.eth.get_block('latest')['timestamp']
print(f"Before fast forward: {now}")
eth_tester.mine_blocks(1000)
now = w3.eth.get_block('latest')['timestamp']
print(f"After fast forward: {now}")
deadline = staking_contract.functions.deadline().call()
print(f"Deadline: {deadline}")
# Step 9: Each account unstakes their tokens and prints the amounts
print("\nUnstaking attempts:")
for account in [a1, a2, a3]:
    tx_hash = staking_contract.functions.unstake().transact({'from': account})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    # Get the new balance
    balance = erc20_contract.functions.balanceOf(account).call()
    print(f"Account {account} unstaked {amount}, new balance is {balance}")
contract_balance = erc20_contract.functions.balanceOf(staking_contract.address).call()
print(f"Remaining balance: {contract_balance}")
