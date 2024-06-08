def log2(x):
    assert x & (x-1) == 0
    return x.bit_length() - 1

def point_add(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    return (
        x1 * x2 - y1 * y2,
        x1 * y2 + x2 * y1
    )

def point_double(pt):
    x1, y1 = pt
    return (2 * x1 * x1 - 1, 2 * x1 * y1)

def point_multiply(pt, n):
    if n == 0:
        return (pt.__class__(1), pt.__class__(0))
    elif n == 1:
        return pt
    else:
        half = point_multiply(pt, n//2)
        o = point_double(half)
        if n % 2:
            return point_add(o, pt)
        else:
            return o

def get_generator(field):
    modulus = field(0).modulus
    for y in range(2, modulus):
        Y_pt = field(y)
        X_pt = field(1-y**2).sqrt()
        point = X_pt, Y_pt
        for _ in range(log2(modulus + 1) - 1):
            point = point_double(point)
        if point != (1, 0):
            return (X_pt, Y_pt)
    raise Exception("Could not find generator")

def get_initial_domain_of_size(field, size):
    assert size < field(0).modulus
    G = get_generator(field)
    for i in range(log2((field.modulus + 1) // size) - 1):
        G = point_double(G)
    Gx2 = point_double(G)
    o = [G]
    for i in range(1, size):
        o.append(point_add(o[-1], Gx2))
    return o

def get_single_domain_value(field, size, index):
    assert size < field(0).modulus
    G = get_generator(field)
    for i in range(log2((field.modulus + 1) // size) - 1):
        G = point_double(G)
    return point_multiply(G, 2*index+1)

def halve_domain(domain, preserve_length=False):
    new_length = len(domain) if preserve_length else len(domain)//2
    if isinstance(domain[0], tuple):
        return [x[0] for x in domain[:new_length]]
    else:
        return [2*x**2-1 for x in domain[:new_length]]

def halve_single_domain_value(value):
    if isinstance(value, tuple):
        return value[0]
    else:
        return 2*value**2-1

def fft(vals, domain=None):
    if len(vals) == 1:
        return vals
    if domain is None:
        domain = get_initial_domain_of_size(vals[0].__class__, len(vals))
    half_domain = halve_domain(domain)
    if isinstance(domain[0], tuple):
        left = vals[:len(domain)//2]
        right = vals[len(domain)//2:][::-1]
        f0 = [(L+R)/2 for L,R in zip(left, right)]
        f1 = [(L-R)/(2*y) for L,R,(x,y) in zip(left, right, domain)]
    else:
        left = vals[:len(domain)//2]
        right = vals[len(domain)//2:][::-1]
        f0 = [(L+R)/2 for L,R in zip(left, right)]
        f1 = [(L-R)/(2*x) for L,R,x in zip(left, right, domain)]
    o = [0] * len(domain)
    o[::2] = fft(f0, half_domain)
    o[1::2] = fft(f1, half_domain)
    return o

def inv_fft(vals, domain=None):
    if len(vals) == 1:
        #print('o', vals)
        return vals
    if domain is None:
        domain = get_initial_domain_of_size(vals[0].__class__, len(vals))
    half_domain = halve_domain(domain)
    f0 = inv_fft(vals[::2], half_domain)
    f1 = inv_fft(vals[1::2], half_domain)
    if isinstance(domain[0], tuple):
        left = [L+y*R for L,R,(x,y) in zip(f0, f1, domain)]
        right = [L-y*R for L,R,(x,y) in zip(f0, f1, domain)]
    else:
        left = [L+x*R for L,R,x in zip(f0, f1, domain)]
        right = [L-x*R for L,R,x in zip(f0, f1, domain)]
    return left+right[::-1]

