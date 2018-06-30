def _fft(vals, modulus, roots_of_unity):
    if len(vals) == 1:
        return vals
    L = _fft(vals[::2], modulus, roots_of_unity[::2])
    R = _fft(vals[1::2], modulus, roots_of_unity[::2])
    o = [0 for i in vals]
    for i, (x, y) in enumerate(zip(L, R)):
        y_times_root = y*roots_of_unity[i]
        o[i] = (x+y_times_root) % modulus 
        o[i+len(L)] = (x-y_times_root) % modulus 
    # print(vals, root_of_unity, o)
    return o

def fft(vals, modulus, root_of_unity, inv=False):
    # Build up roots of unity
    rootz = [1, root_of_unity]
    while rootz[-1] != 1:
        rootz.append((rootz[-1] * root_of_unity) % modulus)
    # Fill in vals with zeroes if needed
    if len(rootz) > len(vals) + 1:
        vals = vals + [0] * (len(rootz) - len(vals) - 1)
    if inv:
        # Inverse FFT
        invlen = pow(len(vals), modulus-2, modulus)
        return [(x*invlen) % modulus for x in _fft(vals, modulus, rootz[::-1])]
    else:
        # Regular FFT
        return _fft(vals, modulus, rootz)

def mul_polys(a, b, modulus, root_of_unity):
    x1 = fft(a, modulus, root_of_unity)
    x2 = fft(b, modulus, root_of_unity)
    return fft([(v1*v2)%modulus for v1,v2 in zip(x1,x2)],
               modulus, root_of_unity, inv=True)

def div_polys(a, b, modulus, root_of_unity):
    x1 = fft(a, modulus, root_of_unity)
    x2 = fft(b, modulus, root_of_unity)
    return fft([(v1*pow(v2,modulus-2,modulus))%modulus for v1,v2 in zip(x1,x2)],
               modulus, root_of_unity, inv=True)
