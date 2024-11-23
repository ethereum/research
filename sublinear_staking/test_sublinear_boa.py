# Copied and edited from
# https://github.com/bout3fiddy/ethereum-research/blob/master/sublinear_staking/test_sublinear_boa.py

import boa  # to install: pip install titanoboa
import pytest  # to install: pip install pytest

@pytest.fixture
def accounts():
    return [boa.env.generate_address() for i in range(5)]


@pytest.fixture
def deployer():
    return boa.env.generate_address()


@pytest.fixture
def minter():
    return boa.env.generate_address()


@pytest.fixture
def erc1155_contract():
    return boa.load("./erc1155.vy")


@pytest.fixture()
def cID():
    return 1


@pytest.fixture
def fund_accounts_erc1155(minter, accounts, cID, erc1155_contract):
        for account in accounts[:-1]:
            erc1155_contract.mint(account, cID, 1)
            print(account, "erc1155'd")
        return erc1155_contract

@pytest.fixture
def erc20_contract():
    return boa.load("./erc20.vy")


@pytest.fixture()
def erc20_initial_balance():
    return 10**18


@pytest.fixture
def fund_accounts_erc20(minter, accounts, erc20_initial_balance, erc20_contract):
        for account in accounts:
            erc20_contract.mint(account, erc20_initial_balance)
            print(account, "erc20'd")
        return erc20_contract


@pytest.fixture
def staking_contract(accounts, erc20_contract, fund_accounts_erc1155, cID, deployer, erc20_initial_balance):
    with boa.env.prank(deployer):
        contract = boa.load(
            "./code.vy", 
            erc20_contract.address, 
            fund_accounts_erc1155.address,
            cID,    
            1,
        )
        
    # handle approvals
    for account in accounts:
        erc20_contract.approve(contract.address, erc20_initial_balance, sender=account)
        
    return contract
        
        
@pytest.fixture
def amount_minted_to_staking_contract():
    return 10**17
        
        
@pytest.fixture
def funded_staking_contract(erc20_contract, staking_contract, minter, amount_minted_to_staking_contract):
    erc20_contract.mint(staking_contract, amount_minted_to_staking_contract, sender=minter)
    return staking_contract


@pytest.fixture
def stake_amounts(accounts):
    return {
        accounts[0]: 1,
        accounts[1]: 10**9,
        accounts[2]: 10**18,
        accounts[3]: 10**18,
        accounts[4]: 10**18,
    }


@pytest.fixture
def test_staking_success(accounts, stake_amounts, funded_staking_contract, fund_accounts_erc20):
    failing_account = accounts[-1]
    for account, amount in stake_amounts.items():
        if account != failing_account:
            print(account, fund_accounts_erc20.balanceOf(account), amount)
            funded_staking_contract.stake(amount, sender=account)

    return funded_staking_contract


def test_staking_fail(accounts, stake_amounts, test_staking_success):
    staking_contract = test_staking_success
    failing_account = accounts[-1]
    with boa.reverts():
        return staking_contract.stake(stake_amounts[failing_account], sender=failing_account)
    

@pytest.fixture
def fast_forward_1000_blocks(test_staking_success):    
    staking_contract = test_staking_success
    
    fundedUntil = staking_contract.fundedUntil()
    current_timestamp = boa.env.evm.patch.timestamp
    assert 1000 < fundedUntil - current_timestamp < 2000
    
    # travel fowards:
    boa.env.time_travel(blocks=1000, block_delta=1)
    
    # return timestamps (current_timestamp is pre-travel)
    return current_timestamp, boa.env.evm.patch.timestamp
    

@pytest.fixture
def test_unstaking_success_sans_a4(
    accounts, stake_amounts, fast_forward_1000_blocks, test_staking_success, erc20_contract
):
    staking_contract = test_staking_success
    previous_time, current_time = fast_forward_1000_blocks
    timedelta = current_time - previous_time
    
    for account in accounts[:3]:
        staking_contract.unstake(sender=account)
        erc20_balance = erc20_contract.balanceOf(account)
        expected_return = int(stake_amounts[account] ** 0.75) * timedelta
        actual_return = erc20_balance - 10**18
        
        assert 0.99 < actual_return / expected_return < 1.01
        
    return staking_contract, current_time, previous_time


def test_unstaking_post_travel_not_full_payment(test_unstaking_success_sans_a4, accounts, stake_amounts, erc20_contract):
    
    staking_contract, current_time, previous_time = test_unstaking_success_sans_a4
    
    fundedUntil = staking_contract.fundedUntil()
    assert fundedUntil - current_time < 2000
    
    # travel:
    boa.env.time_travel(blocks=2000, block_delta=1)
    
    account = accounts[3]
    
    # attempt unstake:
    staking_contract.unstake(sender=account)

    erc20_balance = erc20_contract.balanceOf(account)
    expected_return = int(stake_amounts[account] ** 0.75) * (fundedUntil - previous_time)
    actual_return = erc20_balance - 10**18
    assert 0.99 < actual_return / expected_return < 1.01
    
    staking_contract_balance = erc20_contract.balanceOf(staking_contract.address)
    assert staking_contract_balance > 0
