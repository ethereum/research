from ec65536 import *

a = 124
b = 8932
c = 12415

assert galois_mul(galois_add(a, b), c) == galois_add(galois_mul(a, c), galois_mul(b, c))

k1 = list(range(10))
k2 = list(range(100, 200))
k3 = mul_polys(k1, k2)
assert div_polys(k3, k1) == k2
assert div_polys(k3, k2) == k1
assert galois_mul(eval_poly_at(k1, 9999), eval_poly_at(k2, 9999)) == \
    eval_poly_at(k3, 9999)
k4 = compose_polys(k1, k2)
assert eval_poly_at(k4, 9998) == eval_poly_at(k1, eval_poly_at(k2, 9998))
