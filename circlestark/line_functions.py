from zorch.m31 import (
    zeros, array, append, add, sub, mul, cp as np,
    mul_ext, modinv_ext, sum as m31_sum, modinv
)
from zorch.m31_circle import Point

from precomputes import sub_domains

from fast_fft import bary_eval

def line_function(P1, P2, domain, is_extended=False):
    if is_extended:
        domain = domain.to_extended()
        P1 = P1.to_extended()
        P2 = P2.to_extended()
        _mul = mul_ext
    else:
        _mul = mul
    a = sub(P2.y, P1.y)
    b = sub(P1.x, P2.x)
    c = sub(_mul(P2.x, P1.y), _mul(P1.x, P2.y))
    return add(add(_mul(a, domain.x), _mul(b, domain.y)), c)

def interpolant(P1, v1, P2, v2, domain, is_extended=False):
    if is_extended:
        domain = domain.to_extended()
        P1 = P1.to_extended()
        P2 = P2.to_extended()
        _mul = mul_ext
        _inv = modinv_ext
        shape_tail = (1,) * (len(v1.shape) - 1) + (4,)
    else:
        _mul = mul
        _inv = modinv
        shape_tail = (1,) * len(v1.shape)
    v1 = v1.reshape((1,)+v1.shape)
    v2 = v2.reshape((1,)+v2.shape)
    x = domain.x.reshape((domain.shape[0],) + shape_tail)
    y = domain.y.reshape((domain.shape[0],) + shape_tail)
    dx = sub(P2.x, P1.x)
    dy = sub(P2.y, P1.y)
    invdist = _inv(add(_mul(dx, dx), _mul(dy, dy)))
    dot = add(
        _mul(sub(x, P1.x), dx),
        _mul(sub(y, P1.y), dy)
    )
    return add(
        v1,
        _mul(sub(v2, v1), _mul(dot, invdist))
    )

def public_args_to_vanish_and_interp(domain_size,
                                     indices,
                                     vals,
                                     is_extended=False,
                                     out_domain=None):
    if is_extended:
        _inv = modinv_ext
        depth = len(vals.shape) - 2
        one = array([1,0,0,0])
        _mul = mul_ext
    else:
        _inv = modinv
        depth = len(vals.shape) - 1
        one = array(1)
        _mul = mul
    assert len(indices) % 2 == 0
    next_power_of_2 = 2**(len(indices)-1).bit_length() * 2
    assert next_power_of_2 < domain_size
    lines = []
    eval_domain = sub_domains[next_power_of_2: next_power_of_2*2]
    if out_domain is not None:
        eval_domain = Point(
            append(eval_domain.x, out_domain.x),
            append(eval_domain.y, out_domain.y)
        )
    vpoly = one.reshape((1,)+one.shape)+zeros((eval_domain.shape[0],)+one.shape)
    points = sub_domains[domain_size + array(indices)]
    if is_extended:
        points = points.to_extended()
    for i in range(0, len(indices), 2):
        lines.append(
            line_function(points[i], points[i+1], eval_domain, is_extended)
        )
        vpoly = _mul(vpoly, lines[-1])
    interp = zeros((eval_domain.shape[0],) + vals.shape[1:])
    for i in range(0, len(indices), 2):
        vpoly_adjusted = (
            _mul(vpoly, _inv(lines[i//2]))
            .reshape((eval_domain.shape[0],) + (1,) * depth + one.shape)
        )
        y1 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i], is_extended)
        y2 = bary_eval(vpoly_adjusted[:next_power_of_2], points[i+1], is_extended)
        I = interpolant(
            points[i], _mul(vals[i], _inv(y1)),
            points[i+1], _mul(vals[i+1], _inv(y2)),
            eval_domain,
            is_extended
        )
        interp = add(interp, _mul(vpoly_adjusted, I))
    if out_domain is not None:
        return vpoly[next_power_of_2:], interp[next_power_of_2:]
    else:
        return vpoly, interp
