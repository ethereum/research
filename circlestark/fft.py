def point_add(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    return (
        x1 * x2 - y1 * y2,
        x1 * y2 + x2 * y1
    )

def point_double(pt):
    x1, y1 = pt
    return (2 * x1 * x1 - 1, 2 * x1 * y1)

def get_generator(field):
    modulus = field(0).modulus
    for x in range(2, modulus):
        X_pt = field(x)
        Y_pt = field(1-x**2).sqrt()
        point = X_pt, Y_pt
        for _ in range(modulus.bit_length()-1):
            point = point_double(point)
        if point != (1, 0):
            return (X_pt, Y_pt)
    raise Exception("Could not find generator")

def get_initial_domain(field):
    G = get_generator(field)
    Gx2 = point_double(G)
    o = [G]
    for i in range(field.modulus//2):
        o.append(point_add(o[-1], Gx2))
    return o

def halve_domain(domain):
    if isinstance(domain[0], tuple):
        assert len(domain) == (domain[0][0].modulus + 1) // 2
        return [x[0] for x in domain[:len(domain)//2]]
    else:
        return [2*x**2-1 for x in domain[:len(domain)//2]]

def fft(vals, domain=None):
    #print('i', vals)
    if len(vals) == 1:
        return vals
    if domain is None:
        domain = get_initial_domain(vals[0].__class__)
        while len(domain) > len(vals):
            domain = domain[::2]
    half_domain = halve_domain(domain)
    if len(vals) == (vals[0].modulus + 1) // 2:
        left = vals[:len(domain)//2]
        right = vals[len(domain)//2:][::-1]
        f0 = [(L+R)/2 for L,R in zip(left, right)]
        f1 = [(L-R)/(2*y) for L,R,(x,y) in zip(left, right, domain)]
    else:
        left = vals[:len(domain)//2]
        right = vals[len(domain)//2:][::-1]
        f0 = [(L+R)/2 for L,R in zip(left, right)]
        f1 = [(L-R)/(2*x) for L,R,x in zip(left, right, domain)]
    return fft(f0, half_domain) + fft(f1, half_domain)

def inv_fft(vals, domain=None):
    if len(vals) == 1:
        #print('o', vals)
        return vals
    if domain is None:
        domain = get_initial_domain(vals[0].__class__)
        while len(domain) > len(vals):
            domain = domain[::2]
    half_domain = halve_domain(domain)
    f0 = inv_fft(vals[:len(domain)//2], half_domain)
    f1 = inv_fft(vals[len(domain)//2:], half_domain)
    if len(vals) == (vals[0].modulus + 1) // 2:
        left = [L+y*R for L,R,(x,y) in zip(f0, f1, domain)]
        right = [L-y*R for L,R,(x,y) in zip(f0, f1, domain)]
    else:
        left = [L+x*R for L,R,x in zip(f0, f1, domain)]
        right = [L-x*R for L,R,x in zip(f0, f1, domain)]
    #print('o', left + right[::-1])
    return left+right[::-1]
