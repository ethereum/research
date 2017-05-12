from py_pairing.optimized_pairing import pairing, neg, G2, G1, multiply, FQ12, curve_order
# from bn128_pairing import pairing, neg, G2, G1, multiply, FQ12, curve_order
import time

a = time.time()
print('Starting tests')
p1 = pairing(G2, G1)
pn1 = pairing(G2, neg(G1))
assert p1 * pn1 == FQ12.one()
print('Pairing check against negative in G1 passed')
np1 = pairing(neg(G2), G1)
assert p1 * np1 == FQ12.one()
assert pn1 == np1
print('Pairing check against negative in G2 passed')
assert p1 ** curve_order == FQ12.one()
print('Pairing output has correct order')
p2 = pairing(G2, multiply(G1, 2))
assert p1 * p1 == p2
print('Pairing bilinearity in G1 passed')
assert p1 != p2 and p1 != np1 and p2 != np1
print('Pairing is non-degenerate')
po2 = pairing(multiply(G2, 2), G1)
assert p1 * p1 == po2
print('Pairing bilinearity in G2 passed')
p3 = pairing(multiply(G2, 27), multiply(G1, 37))
po3 = pairing(G2, multiply(G1, 999))
assert p3 == po3
print('Composite check passed')
print('Total time: %.3f' % (time.time() - a))
