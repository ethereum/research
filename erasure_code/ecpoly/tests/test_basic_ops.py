from ecpoly import PrimeField

f = PrimeField(65537)

k1 = list(range(10))
k2 = list(range(100, 200))
k3 = f.mul_polys(k1, k2)
assert f.div_polys(k3, k1) == k2
assert f.div_polys(k3, k2) == k1
assert (f.eval_poly_at(k1, 9999) * f.eval_poly_at(k2, 9999) - 
        f.eval_poly_at(k3, 9999)) % f.modulus == 0
k4 = f.compose_polys(k1, k2)
assert f.eval_poly_at(k4, 9998) == f.eval_poly_at(k1, f.eval_poly_at(k2, 9998))

print("All passed!")
