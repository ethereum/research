from fft import fft, mul_polys

# Calculates modular inverses [1/values[0], 1/values[1] ...]
def multi_inv(values, modulus):
    partials = [1]
    for i in range(len(values)):
        partials.append(partials[-1] * values[i] % modulus)
    inv = pow(partials[-1], modulus - 2, modulus)
    outputs = [0] * len(values)
    for i in range(len(values), 0, -1):
        outputs[i-1] = partials[i-1] * inv % modulus
        inv = inv * values[i-1] % modulus
    return outputs

# Generates q(x) = poly(k * x)
def p_of_kx(poly, modulus, k):
    o = []
    power_of_k = 1
    for x in poly:
        o.append(x * power_of_k % modulus)
        power_of_k = (power_of_k * k) % modulus
    return o

# Return (x - root**positions[0]) * (x - root**positions[1]) * ...
# possibly with a constant factor offset
def _zpoly(positions, modulus, roots_of_unity):
    # If there are not more than 4 positions, use the naive
    # O(n^2) algorithm as it is faster
    if len(positions) <= 4:
        root = [1]
        for pos in positions:
            x = roots_of_unity[pos]
            root.insert(0, 0)
            for j in range(len(root)-1):
                root[j] -= root[j+1] * x
        return [x % modulus for x in root]
    else:
        # Recursively find the zpoly for even indices and odd
        # indices, operating over a half-size subgroup in each
        # case
        left = _zpoly([x//2 for x in positions if x%2 == 0],
                     modulus, roots_of_unity[::2])
        right = _zpoly([x//2 for x in positions if x%2 == 1],
                     modulus, roots_of_unity[::2])
        invroot = roots_of_unity[-1]
        # Offset the result for the odd indices, and combine
        # the two
        o = mul_polys(left, p_of_kx(right, modulus, invroot),
                         modulus, roots_of_unity[1])
    # Deal with the special case where mul_polys returns zero
    # when it should return x ^ (2 ** k) - 1
    if o == [0] * len(o):
        return [1] + [0] * (len(o) - 1) + [modulus - 1]
    else:
        return o

def zpoly(positions, modulus, root_of_unity):
    # Precompute roots of unity
    rootz = [1, root_of_unity]
    while rootz[-1] != 1:
        rootz.append((rootz[-1] * root_of_unity) % modulus)
    return _zpoly(positions, modulus, rootz[:-1])

def erasure_code_recover(vals, modulus, root_of_unity):
    # Generate the polynomial that is zero at the roots of unity
    # corresponding to the indices where vals[i] is None
    import poly_utils
    z = zpoly([i for i in range(len(vals)) if vals[i] is None],
              modulus, root_of_unity)
    zvals = fft(z, modulus, root_of_unity)

    # Pointwise-multiply (vals filling in zero at missing spots) * z
    # By construction, this equals vals * z
    vals_with_zeroes = [x or 0 for x in vals]
    p_times_z_vals = [x*y % modulus for x,y in zip(vals_with_zeroes, zvals)]
    p_times_z = fft(p_times_z_vals, modulus, root_of_unity, inv=True)

    # Keep choosing k values until the algorithm does not fail
    # Check only with primitive roots of unity
    for k in range(2, modulus):
        if pow(k, (modulus - 1) // 2, modulus) == 1:
            continue
        invk = pow(k, modulus - 2, modulus)
        # Convert p_times_z(x) and z(x) into new polynomials
        # q1(x) = p_times_z(k*x) and q2(x) = z(k*x)
        # These are likely to not be 0 at any of the evaluation points.
        p_times_z_of_kx = [x * pow(k, i, modulus) % modulus
                           for i, x in enumerate(p_times_z)]
        p_times_z_of_kx_vals = fft(p_times_z_of_kx, modulus, root_of_unity)
        z_of_kx = [x * pow(k, i, modulus) for i, x in enumerate(z)]
        z_of_kx_vals = fft(z_of_kx, modulus, root_of_unity)
        
        # Compute q1(x) / q2(x) = p(k*x)
        inv_z_of_kv_vals = multi_inv(z_of_kx_vals, modulus)
        p_of_kx_vals = [x*y % modulus for x,y in
                        zip(p_times_z_of_kx_vals, inv_z_of_kv_vals)]
        p_of_kx = fft(p_of_kx_vals, modulus, root_of_unity, inv=True)

        # Given q3(x) = p(k*x), recover p(x)
        p_of_x = [x * pow(invk, i, modulus) % modulus
                  for i, x in enumerate(p_of_kx)]
        output = fft(p_of_x, modulus, root_of_unity)

        # Check that the output matches the input
        success = True
        for inpd, outd in zip(vals, output):
            success *= (inpd is None or inpd == outd)
        if not success:
            continue

        # Output the evaluations if all good
        return output
