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
    
if __name__ == '__main__':
    test_zpoly()
    test_recovery()
