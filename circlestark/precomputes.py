from utils import (
    reverse_bit_order, folded_reverse_bit_order, log2
)
from zorch.m31 import (
    zeros, array, arange, append, tobytes, add, sub, mul, cp as np,
    mul_ext, modinv, modinv_ext, sum as m31_sum, M31
)
from zorch.m31_circle import Point, ExtendedPoint, Z, G

TOP_DOMAIN_SIZE = 2**24

# Generator point
for i in range(log2(TOP_DOMAIN_SIZE), log2(M31+1)-1):
    G = G.double()

def get_subdomains():
    # Compute the points in the largest-size domain
    top_domain = Point.zeros(2)
    top_domain[1] = G
    for i in range(1, log2(TOP_DOMAIN_SIZE * 2)):
        doubled = top_domain.double()
        doubled_plus_one = doubled + G
        new_domain = Point.zeros(2**(i+1))
        new_domain[::2] = doubled
        new_domain[1::2] = doubled_plus_one
        top_domain = new_domain
    top_domain = top_domain[1::2]
    
    # Compute an array that contains the top domain and all smaller-size domains
    sub_domains = Point.zeros(TOP_DOMAIN_SIZE * 2)
    sub_domains[TOP_DOMAIN_SIZE:] = top_domain
    for i in range(log2(TOP_DOMAIN_SIZE)-1, -1, -1):
        sub_domains[2**i:2**(i+1)] = sub_domains[2**(i+1):(2**i)*3].double()
    return sub_domains

sub_domains = get_subdomains()

invx = modinv(sub_domains.x)
invy = modinv(sub_domains.y)

rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    rbos[2**i:2**(i+1)] = reverse_bit_order(arange(2**i))

folded_rbos = zeros(TOP_DOMAIN_SIZE * 2)
for i in range(log2(TOP_DOMAIN_SIZE)):
    folded_rbos[2**i:2**(i+1)] = folded_reverse_bit_order(arange(2**i))
