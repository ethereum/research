from utils import (
    np, array, zeros, tobytes, arange, append, log2, point_add, point_double,
    modinv, one, M31, reverse_bit_order, folded_reverse_bit_order
)

TOP_DOMAIN_SIZE = 2**24

# Generator point
G = array([1268011823, 2])
for i in range(log2(TOP_DOMAIN_SIZE), log2(M31+1)-1):
    G = point_double(G)

# Compute the points in the largest-size domain
top_domain = zeros((2,) + G.shape)
top_domain[0][0] = 1
top_domain[1] = G
for i in range(1, log2(TOP_DOMAIN_SIZE * 2)):
    new_domain = zeros((2**(i+1),) + G.shape)
    new_domain[::2] = point_double(top_domain)
    new_domain[1::2] = point_add(G, new_domain[::2])
    top_domain = new_domain
top_domain = top_domain[1::2]

# Compute an array that contains the top domain and all smaller-size domains
sub_domains = zeros((TOP_DOMAIN_SIZE*2, 2))
sub_domains[TOP_DOMAIN_SIZE:] = top_domain
for i in range(log2(TOP_DOMAIN_SIZE)-1, -1, -1):
    sub_domains[2**i:2**(i+1)] = point_double(sub_domains[2**(i+1):(2**i)*3])

invx = modinv(sub_domains[:,0])
invy = modinv(sub_domains[:,1])

rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    rbos[2**i:2**(i+1)] = reverse_bit_order(arange(2**i))

folded_rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    folded_rbos[2**i:2**(i+1)] = folded_reverse_bit_order(arange(2**i))
