import math

# Length of an epoch in seconds
epoch_len = 1400
# In-protocol penalization parameter
increment = 0.00002

# Parameters
NFP = 0
NCP = 3
NCCP = 3
NPP = 3
NPCP = 3

def sim_offline(p):
    online, offline = 1-p, p
    for i in range(1, 999999):
        # Lost by offline validators
        offline_loss = NFP + NPP + NPCP * offline
        # Lost by online validators
        online_loss = NFP + NPCP * offline
        online *= 1 - increment * math.log(i) * online_loss
        offline *= 1 - increment * math.log(i) * offline_loss
        if i % 100 == 0 or online >= 2 * offline:
            print("%d epochs (%.2f days): online %.3f offline %.3f" %
                  (i, epoch_len * i / 86400, online, offline))
            # If the remaining validators can commit, break
            if online >= 2 * offline:
                break

sim_offline(0.4)
