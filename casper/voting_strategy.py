# The voting strategy. Validators see what every other validator's most
# recent vote for very particular block, in the format
#
# {
#    blockhash1: [vote1, vote2, vote3...],
#    blockhash2: [vote1, vote2, vote3...],
#    ...
# }
#
# Where the votes are probabilities with 0 < p < 1 (see
# http://lesswrong.com/lw/mp/0_and_1_are_not_probabilities/ !), and the
# strategy should itself return an object of the format
# {
#    blockhash1: vote,
#    blockhash2: vote,
#    ...
# }


def vote(probs, db, num_validators):
    pass1 = {k: get_vote_from_scores(v, num_validators)
             for k, v in probs.items() if k in db}
    pass2 = normalize(pass1, num_validators)
    return pass2


# Get the list of scores from other users and come up with
# your own base score
def get_vote_from_scores(probs, num_validators):
    if len(probs) <= num_validators * 2 / 3:
        o1 = 0
    else:
        o1 = sorted(probs)[::-1][num_validators * 2 / 3]
    return 0.8 + 0.2 * o1


# Given a set of independently computed block probabilities, "normalize" the
# probabilities (ie. make sure they sum to at most 1; less than 1 is fine
# because the difference reflects the probability that some as-yet-unknown
# block will ultimately be finalized)
def normalize(block_results, num_validators):
    # Trivial base cases
    if len(block_results) == 1:
        return {k: v for k, v in block_results.items()}
    elif len(block_results) == 0:
        return {}
    a = {k: v for k, v in block_results.items()}
    for v in a.values():
        assert v <= 1, a
    # Artificially privilege the maximum value at the expense of the
    # others; this ensures more rapid convergence toward one equilibrium
    maxkey, maxval = None, 0
    for v in a:
        if a[v] > maxval:
            maxkey, maxval = v, a[v]
    for v in a:
        if v == maxkey:
            a[v] = a[v] * 0.8 + 0.2
        else:
            a[v] *= 0.8
    # If probabilities sum to more than 1, keep reducing them via a
    # transform that preserves proportional probability of non-inclusion
    while 1:
        for v in a.values():
            assert v <= 1, a
        if sum(a.values()) < 1:
            return a
        a = {k: v * 1.05 - 0.050001 for k, v in a.items() if v > 0.050001}
