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
        offline_loss = NFP + NPP + NPCP * (offline / (online + offline))
        # Lost by online validators
        online_loss = NFP + NPCP * (offline / (online + offline))
        online *= 1 - increment * math.log(i) * online_loss
        offline *= 1 - increment * math.log(i) * offline_loss
        if i % 100 == 0 or online >= 2 * offline:
            print("%d epochs (%.2f days): online %.4f offline %.4f" %
                  (i, epoch_len * i / 86400, online, offline))
            # If the remaining validators can commit, break
            if online >= 2 * offline:
                return (1-p, online, epoch_len * i / 86400)

sim_offline(0.4)

#results = [sim_offline(i * 0.01) for i in range(34, 100)]
#for col in results:
#    print("%.4f, %.4f, %.4f" % col)
