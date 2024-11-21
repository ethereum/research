# This is an implementation of a **highly simplified** version of the
# Community Notes algorithm as described in the Birdwatch paper.
# The paper is located at:
# https://github.com/twitter/communitynotes/blob/main/birdwatch_paper_2022_10_27.pdf

import random
import sys

# Randomly generates a table of ratings, where `ratings[i][j]` is the
# i'th user's rating of the j'th note. `size` is the number of users
# and ratings, and `rating_count` is the number of notes that each
# user rates

def generate_ratings(size, rating_count):
    
    # Generate a `size * size` ratings table with the property that
    # each row and each colummn has `rating_count` ratings
    
    ratings = [[0 for _ in range(size)] for _ in range(size)]
    
    perms = []
    while len(perms) < rating_count:
        new_perm = list(range(size))
        random.shuffle(new_perm)
        good = True
        for i in range(size):
            for existing_perm in perms:
                if existing_perm[i] == new_perm[i]:
                    good = False
        if good:
            perms.append(new_perm)
    
        for perm in perms:
            for i, v in enumerate(perm):
                # We model a polarized society, where the top size/2 users
                # and the left size/2 notes are in one tribe, and the bottom
                # size/2 users and the right size/2 notes are in another
                # tribe. By default, users vote +1 on notes in the same tribe
                # as them and -1 on notes in the opposite tribe
                if i%2 == v%2:
                    ratings[i][v] = 1
                else:
                    ratings[i][v] = -1
        
    for i in range(size):
        # The first size/4 notes are "good", so all ratings are increased by 1
        for j in range(size//4):
            if ratings[i][j] != 0:
                ratings[i][j] += 1
        # The second size/4 notes are "good but polarizing"; ratings are
        # increased by 1 but polarization is tripled
        for j in range(size//4, size//2):
            if ratings[i][j] != 0:
                ratings[i][j] *= 3
                ratings[i][j] += 1
        # The third size/4 notes are neutral on quality
        # The fourth size/4 notes are "bad", so all ratings are decreased by 1
        for j in range(size-size//4, size):
            if ratings[i][j] != 0:
                ratings[i][j] -= 1
    return ratings

# Compute the cost function, which determines how much error there is in
# the provided model's ability to model the matrix
def cost(matrix, intercept, user_friendliness, note_quality, user_alignment, note_alignment):
    size = len(matrix)
    total = 0
    for i in range(size):
        for j in range(size):
            expected_value = (
                intercept +
                user_friendliness[i] +
                note_quality[j] +
                user_alignment[i] * note_alignment[j]
            )
            total += (matrix[i][j] - expected_value) ** 2
    # Add regularization params
    total += 0.15 * sum(v**2 for v in ([intercept] * size + user_friendliness + note_quality))
    total += 0.05 * sum(v**2 for v in (user_alignment + note_alignment))
    return total

def average(vals):
    vals = list(vals)
    return sum(vals) / len(vals)

# A really basic descent algorithm that attempts to find values of the model
# that minimize the cost function
def naive_descent(matrix):
    size = len(matrix)
    intercept = [random.randrange(1000) / 1000]
    user_friendliness = [random.randrange(1000) / 1000 for _ in range(size)]
    note_quality = [random.randrange(1000) / 1000 for _ in range(size)]
    user_alignment = [random.randrange(1000) / 1000 for _ in range(size)]
    note_alignment = [random.randrange(1000) / 1000 for _ in range(size)]
    score = cost(
        matrix, intercept[0], user_friendliness, note_quality, user_alignment, note_alignment
    )
    for i in range(9**99):
        round_delta = max(0.001, 0.1 / (i+1)**0.5)
        if i % 50 == 0:
            print("Round {}, cost: {}".format(i, score))
        start_round_score = score

        # Try tweaking each value up and down, and see if that reduces the cost.
        # If it does, keep the change.
        for var in (intercept, user_friendliness, note_quality, user_alignment, note_alignment):
            for i in range(len(var)):
                for delta in (round_delta, -round_delta):
                    var[i] += delta
                    new_score = cost(
                        matrix, intercept[0], user_friendliness,
                        note_quality, user_alignment, note_alignment
                    )
                    if score < new_score:
                        var[i] -= delta
                    else:
                        score = new_score

        # If we can't descend any further, exit
        if score == start_round_score:
            print("Finished descending")
            break

    print("Intercept: {}".format(intercept))
    print("User friendliness: {}".format(user_friendliness))
    print("Note quality: {}".format(note_quality))
    print("User alignment: {}".format(user_alignment))
    print("Note alignment: {}".format(note_alignment))
    print("Group 1 quality: {}".format(average(note_quality[:size//4])))
    print("Group 2 quality: {}".format(average(note_quality[size//4: size//2])))
    print("Group 3 quality: {}".format(average(note_quality[size//2: -size//4])))
    print("Group 4 quality: {}".format(average(note_quality[-size//4:])))
    return note_quality

# Experimental alternative algo
def pairwise_m_tm_m(matrix):
    import numpy as np
    size = len(matrix)
    # Users' votes
    m = np.array(matrix)
    # Basically: 1 if two users agree with each other on-net, 0 otherwise
    user_shared = np.clip(np.dot(m, np.transpose(m)), 0, 1)
    # Likes only
    likes = np.clip(m, 0, 2**30)
    # Dislikes only
    dislikes = -np.clip(m, -2**30, 0)
    # Likes, where a user also "delegates" to everyone who on-net agrees with them
    adj_m_l = np.dot(user_shared, likes)
    sum_of_squares = np.sum(np.square(adj_m_l), axis=1)
    norm_factor = np.sqrt(sum_of_squares)
    adj_m_l = adj_m_l / norm_factor.reshape(-1, 1)
    # Dislikes, where a user also "delegates" to everyone who on-net agrees with them
    adj_m_d = np.dot(user_shared, dislikes)
    sum_of_squares = np.sum(np.square(adj_m_d), axis=1)
    norm_factor = np.sqrt(sum_of_squares)
    adj_m_d = adj_m_d / norm_factor.reshape(-1, 1)
    # Total shared liking between each pair of users
    shared_l = np.dot(adj_m_l, np.transpose(adj_m_l))
    # Total shared disliking between each pair of users
    shared_d = np.dot(adj_m_d, np.transpose(adj_m_d))
    # This is a pairwise algorithm, which walks over each pair of users, and gives them
    # 1 unit of liking power and 1 unit of disliking power to split between notes where
    # they have a common interest. So users who have fewer common interests are counted
    # for more when they do agree on something.
    note_quality = [0] * size
    for i in range(size):
        for j in range(size):
            if shared_l[i, j] > 0: 
                for note in range(size):
                    note_quality[note] += adj_m_l[i, note] * adj_m_l[j, note] / shared_l[i, j]
            if shared_d[i, j] > 0: 
                for note in range(size):
                    note_quality[note] -= adj_m_d[i, note] * adj_m_d[j, note] / shared_d[i, j]
    print("Group 1 quality: {}".format(average(note_quality[:size//4])))
    print("Group 2 quality: {}".format(average(note_quality[size//4: size//2])))
    print("Group 3 quality: {}".format(average(note_quality[size//2: -size//4])))
    print("Group 4 quality: {}".format(average(note_quality[-size//4:])))
    return note_quality


help_string = """
Use:

* --size to specify the number of users and notes
* --votes to specify how many votes each user makes
* --rounds to specify how many rounds to try

Default params are equivalent to:

--size 16 --votes 4 --rounds 1
"""

if __name__ == '__main__':
    SIZE = 16
    ROUNDS = 1
    VOTES = 4
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--size':
            SIZE = int(sys.argv[i+1])
        elif sys.argv[i] == '--rounds':
            ROUNDS = int(sys.argv[i+1])
        elif sys.argv[i] == '--votes':
            VOTES = int(sys.argv[i+1])
        elif sys.argv[i] == '--help':
            print(help_string)
            sys.exit()
    g1_quality, g2_quality, g3_quality, g4_quality = 0, 0, 0, 0
    for i in range(ROUNDS):
        print("Starting round {}".format(i+1))
        ratings = generate_ratings(size=SIZE, rating_count=4)
        print("Randomly generated ratings table:")
        for r in ratings:
            print(r)
        print("Attempting to compute user and note values")
        note_quality = naive_descent(ratings)
        g1_quality += average(note_quality[:SIZE//4])
        g2_quality += average(note_quality[SIZE//4: SIZE//2])
        g3_quality += average(note_quality[SIZE//2: -SIZE//4])
        g4_quality += average(note_quality[-SIZE//4:])
    print("Quality averages:")
    print("Group 1 (good): {}".format(g1_quality / ROUNDS))
    print("Group 2 (good but extra polarizing): {}".format(g2_quality / ROUNDS))
    print("Group 3 (neutral): {}".format(g3_quality / ROUNDS))
    print("Group 4 (bad): {}".format(g4_quality / ROUNDS))
