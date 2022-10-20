import recovery, fft
import random
import time

def test_zpoly():
    for i in range(255):
        indices = []
        for j in range(8):
            if (i >> j) % 2:
                indices.append(j)
        z = recovery.zpoly(indices, 337, 111)
        evals = fft.fft(z, 337, 111)
        for x in range(8):
            if x in indices:
                assert evals[x] == 0
            else:
                assert evals[x] != 0
    print("Passed exhaustive test at size 8")
    modulus = 2**256 - 2**32 * 351 + 1
    nonresidue = 7
    for L in range(3, 13):
        indices = list(set([random.randrange(2**L) for i in range(2**(L-1))]))
        root_of_unity = pow(nonresidue, (modulus-1)//(2**L), modulus)
        a = time.time()
        z = recovery.zpoly(indices, modulus, root_of_unity)
        print('Zpoly of size %d done in time %.4f' % (2**L, time.time() - a))
        evals = fft.fft(z, modulus, root_of_unity)
        for x in range(L):
            if x in indices:
                assert evals[x] == 0
            else:
                assert evals[x] != 0
    print("Passed expansive test")

def test_recovery():
    for i in range(256):
        indices = []
        for j in range(8):
            if (i >> j) % 2:
                indices.append(j)
        if len(indices) > 4:
            continue
        data = fft.fft([random.randrange(337) for i in range(4)], 337, 111)
        erased_data = [data[i] if i not in indices else None
                       for i in range(8)]
        outdata = recovery.erasure_code_recover(erased_data, 337, 111)
        assert data == outdata
    print("Passed exhaustive test at size 8")
    modulus = 2**256 - 2**32 * 351 + 1
    nonresidue = 7
    for L in range(3, 13):
        root_of_unity = pow(nonresidue, (modulus-1)//(2**L), modulus)
        data = fft.fft([random.randrange(modulus) for i in range(2**(L-1))],
                       modulus, root_of_unity)
        indices = {}
        while len(indices) < (2**L * 3 // 8):
            indices[random.randrange(2**L)] = True
        indices = sorted(list(indices))
        erased_data = [data[i] if i not in indices else None
                       for i in range(2**L)]
        a = time.time()
        outdata = recovery.erasure_code_recover(
            erased_data, modulus, root_of_unity
        )
        assert outdata == data
        print("Recovery in dataset of size %i done in time %.4f" %
              (2**L, time.time() - a))
    print("Passed expansive test")

def is_power_of_two(x):
    return x > 0 and x & (x-1) == 0

def reverse_bit_order(n, order):
    """
    Reverse the bit order of an integer n
    """
    assert is_power_of_two(order)
    # Convert n to binary with the same number of bits as "order" - 1, then reverse its bit order
    return int(('{:0' + str(order.bit_length() - 1) + 'b}').format(n)[::-1], 2)


def list_to_reverse_bit_order(l):
    """
    Convert a list between normal and reverse bit order. This operation is idempotent.
    """
    return [l[reverse_bit_order(i, len(l))] for i in range(len(l))]

def test_recovery_danksharding():
    modulus = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    nonresidue = 5
    L = 13 # 2^13 data
    M = 4 # 2^4 sample size

    root_of_unity = pow(nonresidue, (modulus-1)//(2**L), modulus)
    root_of_unity_s = pow(nonresidue, (modulus-1)//(2**(L - M)), modulus)
    root_of_unity_ss = pow(nonresidue, (modulus-1)//(2**M), modulus)
    data = fft.fft([random.randrange(modulus) for i in range(2**(L-1))],
                    modulus, root_of_unity)

    # discard half data
    indices = [i for i in range(2 ** (L - M))] # in samples
    random.shuffle(indices)
    avail_indices=indices[:len(indices)//2]
    missing_indices=indices[len(indices)//2:]
    rbo = list_to_reverse_bit_order([i for i in range(2 ** L)])
    rbo_s = list_to_reverse_bit_order([i for i in range(2 ** (L - M))])
    rbo_ss = list_to_reverse_bit_order([i for i in range(2 ** M)])
    erased_data = data[:]
    for i in missing_indices:
        for j in range(2 ** M):
            erased_data[rbo[i*(2 ** M) + j]] = None

    # recover z in smaller group of order 2 ** (L - M) with shifting parameter
    a = time.time()
    z_s = recovery.zpoly([rbo_s[i] for i in missing_indices],
                modulus, root_of_unity_s)
    # extend to the full group of order 2 ** L
    z = []
    for i in z_s:
        z.append(i)
        z.extend([0] * ((2 ** M) - 1))

    outdata = recovery.erasure_code_recover(
        erased_data, modulus, root_of_unity, z
    )
    assert outdata == data
    print("Recovery in dataset of size %i with optimized zpoly done in time %.4f" %
            (2**L, time.time() - a))
    a = time.time()
    outdata = recovery.erasure_code_recover(
        erased_data, modulus, root_of_unity
    )
    assert outdata == data
    print("Recovery in dataset of size %i done in time %.4f" %
            (2**L, time.time() - a))

    # optimized recovery on danksharding encoding
    # pre-compute root of unit list
    ru_list = [pow(root_of_unity, i, modulus) for i in range(2 ** L)]
    a = time.time()
    # prepare shared variables
    k = 5
    z_of_kx = [x * pow(k, i, modulus) for i, x in enumerate(z_s)]
    z_of_kx_vals = fft.fft(z_of_kx, modulus, root_of_unity_s)
    inv_z_of_kv_vals = recovery.multi_inv(z_of_kx_vals, modulus)
    inv_z_of_kv_vals_map = {k: inv_z_of_kv_vals}
    zvals = fft.fft(z_s, modulus, root_of_unity_s)

    # IFFT all available samples
    coeffs = []
    for i in avail_indices:
        coeff = fft.fft([erased_data[rbo[i * (2 ** M) + j]] for j in rbo_ss], modulus, root_of_unity_ss, True)
        coeff = [c * ru_list[-rbo_s[i] * k] % modulus for k, c in enumerate(coeff)]
        coeffs.extend(coeff)

    # recover pre-FFT values of the missing samples
    recs = []
    for i in range(2 ** M):
        ys = [None] * len(indices)
        for j, y in zip(avail_indices, coeffs[i::(2**M)]):
            ys[rbo_s[j]] = y

        rec = recovery.erasure_code_recover(ys, modulus, root_of_unity_s, z_s, zvals=zvals,
                inv_z_of_kv_vals_map=inv_z_of_kv_vals_map)
        recs.extend(rec)

    # FFT to get the missing samples
    outdata = erased_data[:]
    for i in missing_indices:
        coeff = recs[rbo_s[i]::len(indices)]
        coeff = [c * ru_list[rbo_s[i] * k] % modulus for k, c in enumerate(coeff)]
        outdata_s = fft.fft(coeff, modulus, root_of_unity_ss)
        for j, d in enumerate(outdata_s):
            outdata[rbo[i * (2 ** M) + rbo_ss[j]]] = d
    assert outdata == data
    print("Recovery in dataset of size %i with optimized recovery done in time %.4f" %
            (2**L, time.time() - a))
    print("Passed expansive test")

if __name__ == '__main__':
    test_zpoly()
    test_recovery()
    test_recovery_danksharding()
