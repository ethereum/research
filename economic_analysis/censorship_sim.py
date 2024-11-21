# Total ETH supply
total_supply = 10**8

# Returns to online validators, based on perceived fraction online
def R(p):
    return p**1.53

# Returns to offline validators, based on perceived fraction online
def P(p):
    return 0

# What interest rate are `deposits` ETH worth of validators willing to accept?
def demand_curve(deposits):
    return deposits / total_supply / 10

# Given a total deposit size, and a fraction online, compute the interest
# paid to each online and offline validator
def get_rewards(deposits, p_online):
    # Total txfees
    fees = 50000
    # Portion of fees that get "reclaimed" by the protocol
    reclaimed = 0.5
    # The un-reclaimed fees, which are necessarily distributed among
    # the online validators 
    uncontrolled_reward = fees * (1 - reclaimed) / deposits / p_online
    # The reclaimed fees, minus a portion that gets held back based on
    # the portion of ETH holders staking
    max_controlled_rewards = fees * reclaimed * \
        (deposits / total_supply) ** 0.8
    # In the best possible case (100% online), everyone gets R(1) interest.
    # Rescale interest based on this.
    controlled_rewards_multiplier = max_controlled_rewards / deposits / R(1)
    # Return computed interest
    return uncontrolled_reward + controlled_rewards_multiplier * R(p_online), \
        controlled_rewards_multiplier * P(p_online)

# Total deposits in the validator set
deposits = 1000000

# Find the pre-attack equilibrium
for i in range(100):
    interest, _ = get_rewards(deposits, 1)
    if interest < demand_curve(deposits):
        deposits -= 10000
    else:
        deposits += 10000

attacker = deposits * 0.501
print('Baseline total deposits:', deposits)
print('Baseline interest:', get_rewards(deposits, 1)[0])
print('Baseline attacker revenue:', get_rewards(deposits, 1)[0] * attacker)
print('Baseline victim revenue:', get_rewards(deposits, 1)[0] * (deposits - attacker))

# Start the attack. Find the post-attack equilibrium.
for i in range(1000):
    attacker_share = attacker / deposits
    atk_interest, vic_interest = get_rewards(deposits, attacker_share)
    if vic_interest < demand_curve(deposits):
        deposits -= min(10000, deposits - attacker)
    else:
        deposits += 10000

print('New total deposits:', deposits)
print('Attacker share:', attacker_share)
print('Victim interest:', vic_interest)
print('Attacker interest:', atk_interest)
print('Attacker revenue:', atk_interest * attacker)
print('Possible non-attacking revenue:', get_rewards(deposits, 1)[0])
