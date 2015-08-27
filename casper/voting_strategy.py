import random

# The voting strategy. Validators see what every other validator votes,
# and return their vote.
#
# Votes are probabilities with 0 < p < 1 (see
# http://lesswrong.com/lw/mp/0_and_1_are_not_probabilities/ !)

def vote_transform(p):
    return (abs(2 * p - 1) ** 0.333 * (1 if p > 0.5 else -1) + 1) / 2
    

def vote(probs):
    if len(probs) == 0:
        return 0.5
    probs = sorted(probs)
    score = (probs[len(probs)/2] + probs[-len(probs)/2]) * 0.5
    if score > 0.9:
        score2 = probs[len(probs)/3]
        score = min(score, max(score2, 0.9))
    elif score < 0.1:
        score2 = probs[len(probs)*2/3]
        score = max(score, min(score2, 0.1))
    return vote_transform(score)
