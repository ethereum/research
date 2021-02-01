from math import *
from collections import Counter
from functools import reduce
import bach_random_factored_numbers

lg = lambda x: log(x, 2)

def categorize(tup, easy_ecm_frac=(180./3800), easy_ecm_reject_frac=(180./3800),
               ecm_frac=(621./3800), gnfs_frac=(2027./3800)):
    """
    Categorizes a number into different kinds of moduli:

    easy_ecm_frac: bit fraction of largest factor we can remove by ECM factorization.
    easy_ecm_reject_frac: We reject after this many bits have been "removed" by easy ECM
                factorization.
    ecm_frac: bit fraction of largest factor an attacker could remove by ECM.
    gnfs_frac: bit fraction of largest composite an attacker could factorize by GNFS.

    Output:
        [rejected numbers]
        can_reject: We can reject this modulus because it too much has been factored
        can_factor:  After removing easy ECM factors, a prime is left

        [invalid numbers - could not reject but weak RSA moduli]
        invalid_ecm - invalid because all its factors are smaller than the  threshold defined
                by ecm_frac
        invalid_gnfs - invalid because after removing the ECM factors the remaining rough part
                is under then gnfs_frac threshold

        [valid numbers - good RSA moduli]
        valid_lopsided - valid, second factor less than gnfs_bits / 2
        valid - valid
    """
    n = tup[0]
    f = tup[1][:]
    assert len(f) > 0
    lg_n = lg(n)
    easy_ecm_bits = easy_ecm_frac*lg_n
    easy_ecm_reject_bits = easy_ecm_reject_frac*lg_n
    ecm_bits = ecm_frac*lg_n
    gnfs_bits = gnfs_frac*lg_n
    easy_f = filter(lambda x: lg(x) <= easy_ecm_bits, f)
    if lg(reduce(lambda x,y:x*y, easy_f, 1)) > easy_ecm_reject_bits:
        return 'can_reject'
    f = list(filter(lambda x: lg(x) > easy_ecm_bits, f))
    if len(f) < 2:
        return 'can_factor'
    f = list(filter(lambda x: lg(x) > ecm_bits, f))
    if len(f) < 2:
        return 'invalid_ecm'
    x = reduce(lambda x,y:x*y, f)
    if lg(x) < gnfs_bits:
        return 'invalid_gnfs'
    f.sort()
    if lg(f[-2]) < gnfs_bits/2:
        return 'valid_lopsided'
    return 'valid'

def p_bad(easy_ecm_frac=(180./3800), easy_ecm_reject_frac=0, \
          ecm_frac=(621./3800), gnfs_frac=(2027./3800)):
    """
    Compute approximate probability of getting a bad RSA modulus, using the given
    parameters.
    """
    results = list(map(lambda x: categorize(x, easy_ecm_frac, \
                            easy_ecm_reject_frac, ecm_frac, gnfs_frac), factor_arr))
    non_rejected = list(filter(lambda x: x not in ["can_reject", "can_factor"], results))
    good_moduli = list(filter(lambda x: x in ["valid_lopsided", "valid"], results))
    return 1 - 1.0 * len(good_moduli) / len(non_rejected)

if __name__ == "__main__":
    import numpy as np
    #import matplotlib.pyplot as plt

    # Difficulty 1e12
    # I.e. this aims for moduli which are ca. 1e12 times
    easy_ecm_bits = 180.
    ecm_reject_bits = 0.
    ecm_bits = 621.
    gnfs_bits = 2027.

    # Generate 10,000 128-bit random factored numbers.
    # These numbers are only for quick testing, need a lot more numbers and more bits
    # for accurate results
    factor_arr = bach_random_factored_numbers.generate_random_factored_numbers(128, 8, 10000)

    # Uncomment to load factors from  file instead
    #factor_arr = []
    #with open("random_factored_numbers.txt", "r") as f:
    #    for l in f:
    #        factor_arr.append(eval(l))

    x = np.arange(3000, 5000, 25)
    y = list(map(lambda b: p_bad(easy_ecm_frac=(easy_ecm_bits/b), easy_ecm_reject_frac=(ecm_reject_bits/b), \
              ecm_frac=(ecm_bits/b), gnfs_frac=(gnfs_bits/b)), x))

    #plt.plot(x, y)
    #plt.show()

    y2 = list(map(lambda x: -log(x[0]) / x[1], zip(y, x)))
    #plt.plot(x, y2, label="-log(p)/b")
    #plt.title("RSA moduli difficulti level 1e12")
    #plt.legend()
    #plt.show()

    fun, best_p_1e12, best_b_1e12 = max(zip(y2, y, x))
    print("Best value: p_bad=", best_p_1e12, "at", best_b_1e12, "bits")
    print("Modulus length for 1e-9 chance of bad modulus=", log(1e-9) / log(best_p_1e12) * best_b_1e12)
    print("Modulus length for 1e-12 chance of bad modulus=", log(1e-12) / log(best_p_1e12) * best_b_1e12)
    print("Modulus length for 1e-15 chance of bad modulus=", log(1e-15) / log(best_p_1e12) * best_b_1e12)
