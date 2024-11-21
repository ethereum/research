from zorch.m31 import (
    M31, ExtendedM31, Point, modulus, zeros_like, Z, G
)
from utils import cp

from precomputes import sub_domains

from fast_fft import bary_eval

def line_function(P1, P2, domain):
    a = P2.y - P1.y
    b = P1.x - P2.x
    c = (P2.x * P1.y) - (P1.x * P2.y)
    return a * domain.x + b * domain.y + c

def interpolant(P1, v1, P2, v2, domain):
    v1 = v1.reshape((1,)+v1.shape)
    v2 = v2.reshape((1,)+v2.shape)
    x = domain.x.reshape((domain.shape[0],) + (1,) * (len(v1.shape) - 1))
    y = domain.y.reshape((domain.shape[0],) + (1,) * (len(v1.shape) - 1))
    dx = P2.x - P1.x
    dy = P2.y - P1.y
    invdist = 1 / (dx * dx + dy * dy)
    dot = (x - P1.x) * dx + (y - P1.y) * dy
    return v1 + (v2 - v1) * dot * invdist

def public_args_to_vanish_and_interp(domain_size,
                                     indices,
                                     vals,
                                     out_domain=None):
    assert len(indices) % 2 == 0
    next_power_of_2 = 2**(len(indices)-1).bit_length() * 2
    assert next_power_of_2 < domain_size
    lines = []
    eval_domain = sub_domains[next_power_of_2: next_power_of_2*2]
    if out_domain is not None:
        eval_domain = Point(
            M31.append(eval_domain.x, out_domain.x),
            M31.append(eval_domain.y, out_domain.y)
        )
    vpoly = M31.zeros(eval_domain.shape) + 1
    points = sub_domains[domain_size + cp.array(indices)]
    for i in range(0, len(indices), 2):
        lines.append(
            line_function(points[i], points[i+1], eval_domain)
        )
        vpoly = vpoly * lines[-1]
    interp = vpoly.__class__.zeros((eval_domain.shape[0],) + vals.shape[1:])
    for i in range(0, len(indices), 2):
        vpoly_adjusted = (
            (vpoly / lines[i//2])
            .reshape((eval_domain.shape[0],) + (1,) * (len(vals.shape) - 1))
        )
        y1 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i])
        y2 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i+1])
        I = interpolant(
            points[i], vals[i] / y1,
            points[i+1], vals[i+1] / y2,
            eval_domain,
        )
        interp += vpoly_adjusted * I
    if out_domain is not None:
        return vpoly[next_power_of_2:], interp[next_power_of_2:]
    else:
        return vpoly, interp
