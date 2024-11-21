import binary_fft

bigf = binary_fft.BinaryField(1033)
poly = [x**9 % 1024 for x in range(1024)]
z = binary_fft.fft(bigf, range(1024), poly)
z2 = binary_fft._simple_ft(bigf, range(1024), poly)
assert z == z2
poly2 = binary_fft.invfft(bigf, range(1024), z)
assert poly2 == poly
print("Invfft and fft tests passed")
poly3 = [x**9 % 1024 for x in range(25)]
xs = [x*11 % 32 for x in range(25)]
ys = [bigf.eval_poly_at(poly3, x) for x in xs]
poly4 = binary_fft.interpolate(bigf, xs, ys)
assert poly4[:len(poly3)] == poly3
xs = [x*11 % 32 for x in range(1, 25)]
ys = [bigf.eval_poly_at(poly3, x) for x in xs]
poly5 = binary_fft.interpolate(bigf, xs, ys)
assert poly5[:len(poly3)] == poly3
print("Interpolation tests passed")
