NUMSIGNERS = 15
ATTACKER_SHARE = 0.495
CHANCE_OF_SUCCESS = 0.049
SCORE_DIFFERENTIAL = 10
ATTACKER_VOTE = 0.95
import random

def sim():
    d = -SCORE_DIFFERENTIAL * 15
    while d < 0 and d > -(SCORE_DIFFERENTIAL * 15)-1000:
        if random.random() < ATTACKER_SHARE:
            for i in range(NUMSIGNERS):
                if random.random() < ATTACKER_SHARE:
                    d += ATTACKER_VOTE
                else:
                    d += min(CHANCE_OF_SUCCESS, 0.95)
        else:
            for i in range(NUMSIGNERS):
                if random.random() < ATTACKER_SHARE:
                    pass
                else:
                    d -= min(1 - CHANCE_OF_SUCCESS, 0.95)
    return 1 if d >= 0 else 0
