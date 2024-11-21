import blst
from itertools import zip_longest
from collections import defaultdict
from random import randint
from time import time

def integer_in_base(i, b):
    r = []
    while i > 0:
        r.append(i % b)
        i //= b
    return r

def pippenger_simple(group_elements, factors):
    """
    A naive implementation of a Pippenger-like multiexponentiation algorithm. Don't use this
    in practice, a native implementation in the blst library will perform much better.
    """
    assert len(group_elements) == len(factors)
    n = len(group_elements)
    d = 1
    while (d + 2) * 2**(d + 2) < n:
        d += 1
    b = 2**d
    factors_decomposed = [integer_in_base(factor, b) for factor in factors]
    result = blst.P1_generator().mult(0)
    for bases in reversed(list(zip_longest(*factors_decomposed, fillvalue=0))):
        total = blst.P1_generator().mult(0)
        base_elements_dict = defaultdict(list)
        for index, base in enumerate(bases):
            if base > 0:
                base_elements_dict[base].append(group_elements[index])
        for base, base_elements in base_elements_dict.items():
            if len(base_elements) > 0:
                sum_base_elements = base_elements[0].dup()
                for x in base_elements[1:]:
                    sum_base_elements.add(x)
                sum_base_elements.mult(base)
                total.add(sum_base_elements)
        result.mult(b).add(total)
    return result

def lincomb_naive(group_elements, factors):
    """
    Direct linear combination
    """
    assert len(group_elements) == len(factors)
    result = blst.P1_generator().mult(0)
    for g, f in zip(group_elements, factors):
        result.add(g.dup().mult(f))
    return result

def test_pippenger(group_elements, factors):
    """
    Test and time pippenger_simple
    """
    time_a = time()
    naive_result = lincomb_naive(group_elements, factors)
    time_b = time()
    print("n = {0} multiexp".format(len(group_elements)))
    print("Naive linear combination: {0:.6f} s".format(time_b - time_a))
    pippenger_result = pippenger_simple(group_elements, factors)
    time_c = time()
    print("Using simple Pippenger algorithm: {0:.6f} s".format(time_c - time_b))
    assert naive_result.is_equal(pippenger_result)
    
if __name__ == "__main__":
    test_pippenger([blst.P1_generator()]*16384, [randint(0, 2**255) for i in range(16384)])