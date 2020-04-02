def _simple_ft(vals, modulus, roots_of_unity):
    L = len(roots_of_unity)
    o = []
    for i in range(L):
        last = 0
        for j in range(L):
            last += vals[j] * roots_of_unity[(i*j)%L]
        o.append(last % modulus)
    return o

def _fft(vals, modulus, roots_of_unity):
    if len(vals) <= 4:
        #return vals
        return _simple_ft(vals, modulus, roots_of_unity)
    L = _fft(vals[::2], modulus, roots_of_unity[::2])
    R = _fft(vals[1::2], modulus, roots_of_unity[::2])
    o = [0 for i in vals]
    for i, (x, y) in enumerate(zip(L, R)):
        y_times_root = y*roots_of_unity[i]
        o[i] = (x+y_times_root) % modulus 
        o[i+len(L)] = (x-y_times_root) % modulus 
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
        return [(x*invlen) % modulus for x in
                _fft(vals, modulus, rootz[:0:-1])]
    else:
        # Regular FFT
        return _fft(vals, modulus, rootz[:-1])

# Evaluates f(x) for f in evaluation form
def inv_fft_at_point(vals, modulus, root_of_unity, x):
    if len(vals) == 1:
        return vals[0]
    # 1/2 in the field
    half = (modulus + 1)//2
    # 1/w
    inv_root = pow(root_of_unity, len(vals)-1, modulus)
    # f(-x) in evaluation form
    f_of_minus_x_vals = vals[len(vals)//2:] + vals[:len(vals)//2]
    # e(x) = (f(x) + f(-x)) / 2 in evaluation form
    evens = [(f+g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    # o(x) = (f(x) - f(-x)) / 2 in evaluation form
    odds = [(f-g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    # e(x^2) + coordinate * x * o(x^2) in evaluation form
    comb = [(o * x * inv_root**i + e) % modulus for i, (o, e) in enumerate(zip(odds, evens))]
    return inv_fft_at_point(comb[:len(comb)//2], modulus, root_of_unity ** 2 % modulus, x**2 % modulus)

def shift_domain(vals, modulus, root_of_unity, factor):
    if len(vals) == 1:
        return vals
    # 1/2 in the field
    half = (modulus + 1)//2
    # 1/w
    inv_factor = pow(factor, modulus - 2, modulus)
    half_length = len(vals)//2
    # f(-x) in evaluation form
    f_of_minus_x_vals = vals[half_length:] + vals[:half_length]
    # e(x) = (f(x) + f(-x)) / 2 in evaluation form
    evens = [(f+g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    print('e', evens)
    # o(x) = (f(x) - f(-x)) / 2 in evaluation form
    odds = [(f-g) * half % modulus for f,g in zip(vals, f_of_minus_x_vals)]
    print('o', odds)
    shifted_evens = shift_domain(evens[:half_length], modulus, root_of_unity ** 2 % modulus, factor ** 2 % modulus)
    print('se', shifted_evens)
    shifted_odds = shift_domain(odds[:half_length], modulus, root_of_unity ** 2 % modulus, factor ** 2 % modulus)
    print('so', shifted_odds)
    return (
        [(e + inv_factor * o) % modulus for e, o in zip(shifted_evens, shifted_odds)] + 
        [(e - inv_factor * o) % modulus for e, o in zip(shifted_evens, shifted_odds)]
    )

def shift_poly(poly, modulus, factor):
    factor_power = 1
    inv_factor = pow(factor, modulus - 2, modulus)
    o = []
    for p in poly:
        o.append(p * factor_power % modulus)
        factor_power = factor_power * inv_factor % modulus
    return o

def mul_polys(a, b, modulus, root_of_unity):
    rootz = [1, root_of_unity]
    while rootz[-1] != 1:
        rootz.append((rootz[-1] * root_of_unity) % modulus)
    if len(rootz) > len(a) + 1:
        a = a + [0] * (len(rootz) - len(a) - 1)
    if len(rootz) > len(b) + 1:
        b = b + [0] * (len(rootz) - len(b) - 1)
    x1 = _fft(a, modulus, rootz[:-1])
    x2 = _fft(b, modulus, rootz[:-1])
    return _fft([(v1*v2)%modulus for v1,v2 in zip(x1,x2)],
               modulus, rootz[:0:-1])
