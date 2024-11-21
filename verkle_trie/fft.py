import blst

def _fft(vals, modulus, roots_of_unity):
    if len(vals) == 1:
        return vals
    L = _fft(vals[::2], modulus, roots_of_unity[::2])
    R = _fft(vals[1::2], modulus, roots_of_unity[::2])
    if isinstance(vals[0], blst.P1):
        o = [blst.G1().mult(0) for i in vals]
    else: 
        o = [0 for i in vals]
    for i, (x, y) in enumerate(zip(L, R)):
        y_times_root = y.dup().mult(roots_of_unity[i]) if isinstance(vals[0], blst.P1) else y * roots_of_unity[i]
        o[i] = x.dup().add(y_times_root) if isinstance(vals[0], blst.P1) else (x + y_times_root) % modulus
        o[i + len(L)] = x.dup().add(y_times_root.neg()) if isinstance(vals[0], blst.P1) else (x - y_times_root) % modulus
    return o

def expand_root_of_unity(root_of_unity, modulus):
    # Build up roots of unity
    rootz = [1, root_of_unity]
    while rootz[-1] != 1:
        rootz.append((rootz[-1] * root_of_unity) % modulus)
    return rootz

def fft(vals, modulus, root_of_unity, inv=False):
    rootz = expand_root_of_unity(root_of_unity, modulus)
    # Fill in vals with zeroes if needed
    if len(rootz) > len(vals) + 1:
        vals = vals + [0] * (len(rootz) - len(vals) - 1)
    if inv:
        # Inverse FFT
        invlen = pow(len(vals), modulus-2, modulus)
        if isinstance(vals[0], blst.P1):
            return [x.dup().mult(invlen) for x in
                    _fft(vals, modulus, rootz[:0:-1])]
        else:
            return [(x * invlen) % modulus for x in
                    _fft(vals, modulus, rootz[:0:-1])]
    else:
        # Regular FFT
        return _fft(vals, modulus, rootz[:-1])