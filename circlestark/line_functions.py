from utils import (
    M31, M31SQ, modinv_ext, to_ext_if_needed, modinv,
    zeros, append, array, np
)

from precomputes import sub_domains

from fast_fft import bary_eval

def line_function(P1, P2, domain, arith):
    one, add, mul = arith
    if one.ndim == 1:
        domain = to_ext_if_needed(domain, object_dim=2)
        P1 = to_ext_if_needed(P1, object_dim=1)
        P2 = to_ext_if_needed(P2, object_dim=1)
    a = (P2[1] + M31 - P1[1]) % M31
    b = (P1[0] + M31 - P2[0]) % M31
    c = (M31SQ + mul(P2[0], P1[1]) - mul(P1[0], P2[1])) % M31
    return (mul(a, domain[:,0]) + mul(b, domain[:,1]) + c) % M31

def interpolant(P1, v1, P2, v2, domain, arith):
    one, add, mul = arith
    depth = len(v1.shape) - len(one.shape)
    inv = modinv_ext if one.ndim == 1 else modinv
    v1 = v1.reshape((1,)+v1.shape)
    v2 = v2.reshape((1,)+v2.shape)
    if one.ndim == 1:
        domain = to_ext_if_needed(domain, object_dim=2)
        P1 = to_ext_if_needed(P1, object_dim=1)
        P2 = to_ext_if_needed(P2, object_dim=1)
    x = domain[:,0].reshape((domain.shape[0],) + (1,) * depth + one.shape)
    y = domain[:,1].reshape((domain.shape[0],) + (1,) * depth + one.shape)
    dx = (P2[0] - P1[0]) % M31
    dy = (P2[1] - P1[1]) % M31
    invdist = inv((mul(dx, dx) + mul(dy, dy)) % M31)
    dot = (mul((x - P1[0]) % M31, dx) + mul((y - P1[1]) % M31, dy)) % M31
    return v1 + mul((v2 - v1) % M31, mul(dot, invdist))

def public_args_to_vanish_and_interp(domain_size,
                                     indices,
                                     vals,
                                     arith,
                                     out_domain=None):
    one, add, mul = arith
    inv = modinv_ext if one.ndim == 1 else modinv
    assert len(indices) % 2 == 0
    next_power_of_2 = 2**(len(indices)-1).bit_length() * 2
    assert next_power_of_2 < domain_size
    depth = len(vals.shape) - 1 - one.ndim
    lines = []
    eval_domain = sub_domains[next_power_of_2: next_power_of_2*2]
    if out_domain is not None:
        eval_domain = append(eval_domain, out_domain)
    vpoly = one.reshape((1,)+one.shape)+zeros((eval_domain.shape[0],)+one.shape)
    points = sub_domains[domain_size + array(indices)]
    if one.ndim == 1:
        points = to_ext_if_needed(points, object_dim=2)
    for i in range(0, len(indices), 2):
        lines.append(
            line_function(points[i], points[i+1], eval_domain, arith)
        )
        vpoly = mul(vpoly, lines[-1])
    interp = zeros((eval_domain.shape[0],) + vals.shape[1:])
    for i in range(0, len(indices), 2):
        vpoly_adjusted = (
            mul(vpoly, inv(lines[i//2]))
            .reshape((eval_domain.shape[0],) + (1,) * depth + one.shape)
        )
        y1 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i], arith)
        y2 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i+1], arith)
        I = interpolant(
            points[i], mul(vals[i], inv(y1)),
            points[i+1], mul(vals[i+1], inv(y2)),
            eval_domain,
            arith
        )
        interp += mul(vpoly_adjusted, I)
    if out_domain is not None:
        return vpoly[next_power_of_2:], interp[next_power_of_2:] % M31
    else:
        return vpoly, interp % M31
