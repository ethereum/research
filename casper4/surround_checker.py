# NO_SURROUND validation checker. For any attestation, call `new_attestation(s, t)`.
# If there is a no surround validation, you will be notified. Requires 2*N*log(N)
# bits per validator in addition to the attestations themselves.

class CollisionFound(Exception):
    pass

class Checker():
    def __init__(self, MAX=16):
        self.array1 = [i for i in range(MAX)]
        self.array2 = [MAX for i in range(MAX)]
        self.attestations = []

    def new_attestation(self, s, t):
        assert s < t
        if s > 0 and t < self.array1[s - 1]:
            for os, ot in self.attestations:
                if os < s < t < ot:
                    raise CollisionFound("Collision found: ({} {}) surrounds provided ({} {})".format(os, ot, s, t))
            raise Exception("panic, should never be here")
        if t > self.array2[s + 1]:
            for os, ot in self.attestations:
                if s < os < ot < t:
                    raise CollisionFound("Collision found: ({} {}) is surrounded by provided ({} {})".format(os, ot, s, t))
            raise Exception("panic, should never be here")
        _s = s
        while self.array1[_s] < t and _s < t:
            self.array1[_s] = t
            _s += 1
        _s = s
        while self.array2[_s] > t and _s >= 0:
            self.array2[_s] = t
            _s -= 1
        for os, ot in self.attestations:
            assert not((os < s < t < ot) or (s < os < ot < t))
        self.attestations.append((s, t))

def test():
    c = Checker(16)
    import random
    for i in range(30):
        x, y = random.randrange(16), random.randrange(16)
        if x == y:
            continue
        try:
            c.new_attestation(min(x, y), max(x, y))
        except CollisionFound as e:
            print(e)
    print("Test successful")

if __name__ == '__main__':
    test()
