import poly_utils as binary_fft

bigf = binary_fft.BinaryField(1033)
poly = [x**9 % 1024 for x in range(1024)]
z = binary_fft.fft(bigf, poly, range(1024))
z2 = binary_fft._simple_ft(bigf, poly)
assert z == z2
poly2 = binary_fft.invfft(bigf, z, range(1024))
assert poly2 == poly
print("Tests passed")
