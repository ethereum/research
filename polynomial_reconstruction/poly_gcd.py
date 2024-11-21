from poly_utils import PrimeField
from fft import fft
from math import prod

def next_power_of_two(x):
    return 2**((x - 1).bit_length())

class PrimeFieldExtended(PrimeField):

    def __init__(self, modulus, primitive_root):
        assert pow(2, modulus, modulus) == 2
        self.modulus = modulus
        assert pow(primitive_root, modulus - 1, modulus) == 1
        assert pow(primitive_root, (modulus - 1) // 2, modulus) != 1
        self.primitive_root = primitive_root

    def mul_many_polys(self, ps, result_in_evaluation_form=False, size=0):
        if result_in_evaluation_form:
            n = size
        else:
            n = next_power_of_two(sum(self.degree(p) for p in ps) + 1)
        if (self.modulus - 1) % n == 0:
            root_of_unity = pow(self.primitive_root, (self.modulus - 1) // n, self.modulus)
            ps_fft = [fft(p, self.modulus, root_of_unity) for p in ps]
            r = [prod(l) for l in zip(*ps_fft)]
            if result_in_evaluation_form:
                return r
            return self.truncate_poly(fft(r, self.modulus, root_of_unity, inv=True))
        else:
            assert not result_in_evaluation_form
            return reduce(lambda a, b: super().mul_polys(a, b), ps)

    def mul_polys(self, a, b):
        if self.degree(a) is None or self.degree(b) is None or self.degree(a) < 64 or self.degree(b) < 64:
            return super().mul_polys(a, b)
        else:
            n = next_power_of_two(self.degree(a) + self.degree(b) + 1)
            if (self.modulus - 1) % n == 0:
                root_of_unity = pow(self.primitive_root, (self.modulus - 1) // n, self.modulus)
                assert pow(root_of_unity, n, self.modulus) == 1
                assert pow(root_of_unity, n // 2, self.modulus) != 1
                a_fft = fft(self.truncate_poly(a), self.modulus, root_of_unity)
                b_fft = fft(self.truncate_poly(b), self.modulus, root_of_unity)
                r = [x * y for x, y in zip(a_fft, b_fft)]
                return self.truncate_poly(fft(r, self.modulus, root_of_unity, inv=True))
            else:
                return super().mul_polys(a, b)
            
    
    def degree(self, p):
        for i in range(len(p) - 1, -1, -1):
            if p[i] % self.modulus != 0:
                return i

    def leading_coefficient(self, p):
        if self.degree(p) is None:
            return 1
        return p[self.degree(p)]
    
    def truncate_poly(self, p):
        if self.degree(p) is not None:
            return p[:self.degree(p) + 1]
        return []

    def poly_quotient_remainder(self, p, q):
        if self.degree(p) is None or self.degree(p) < self.degree(q):
            return ([], p)
        else:
            leading_p = self.leading_coefficient(p)
            leading_q = self.leading_coefficient(q)
            factor = self.div(leading_p, leading_q)
            part_quotient = [0] * (self.degree(p) - self.degree(q)) + [self.div(leading_p, leading_q)]
            part_quotient_times_q = [0] * (self.degree(p) - self.degree(q)) + [self.mul(factor, x) for x in q]
            part_remainder = self.sub_polys(p, part_quotient_times_q)
            part2_quotient, remainder = self.poly_quotient_remainder(part_remainder, q)
            return self.truncate_poly(self.add_polys(part_quotient, part2_quotient)), self.truncate_poly(remainder)
    
    
    def multiply_poly_mat_vec(self, A, v):
        assert all(len(row) == len(v) for row in A)
        result = []
        for row in A:
            r = []
            for a, b in zip(row, v):
                r = self.add_polys(r, self.mul_polys(a, b))
            result.append(r)
        return result
    
    def multiply_poly_matrices(self, A, B):
        return list(zip(*[self.multiply_poly_mat_vec(A, x) for x in zip(*B)]))

    def poly_quo(self, p, k):
        if k >= 0:
            return p[k:]
        else:
            return p
    
    def M_hgcd(self, a, b):
        # https://www.csd.uwo.ca/~mmorenom/CS424/Lectures/FastDivisionAndGcd.html/node6.html
        d = self.degree(a)
        if d is None:
            return [[[1],[0]],[[0],[1]]]
        m = (d + 1) // 2
        if self.degree(b) is None or self.degree(b) == 0 or self.degree(b) < m:
            return [[[1],[0]],[[0],[1]]]
        a_quo = self.poly_quo(a, m)
        b_quo = self.poly_quo(b, m)
        M1 = self.M_hgcd(a_quo, b_quo)
        t, s = self.multiply_poly_mat_vec(M1, (a, b))
        if self.degree(s) is None:
            return M1
        q, r = self.poly_quotient_remainder(t, s)
        if self.degree(r) is None:
            M2 = [[[0],[1]],[[1],[-x for x in q]]]
            return self.multiply_poly_matrices(M2, M1)
        v = self.inv(self.leading_coefficient(r))
        r_ = self.mul_polys(r, [v])
        M2 = [[[0],[1]],[[v],self.mul_polys([-v], q)]]
        l = 2 * m - self.degree(s)
        s_quo = self.poly_quo(s, l)
        r_quo = self.poly_quo(r_, l)
        assert self.degree(s_quo) is None or self.degree(r_quo) is None or self.degree(s_quo) >= self.degree(s_quo)
        M3 = self.M_hgcd(s_quo, r_quo)
        return self.multiply_poly_matrices(M3, self.multiply_poly_matrices(M2, M1))
    
    def M_gcd(self, a, b):
        M1 = self.M_hgcd(a, b)
        t, s = self.multiply_poly_mat_vec(M1, [a, b])
        if self.degree(s) is None:
            return M1
        q, r = self.poly_quotient_remainder(t, s)
        if self.degree(r) is None:
            M2 = [[[0],[1]],[[1],[-x for x in q]]]
            return self.multiply_poly_matrices(M2, M1)
        v = self.inv(self.leading_coefficient(r))
        r_ = self.mul_polys(r, [v])
        M2 = [[[0],[1]],[[v],self.mul_polys([-v], q)]]
        M3 = self.M_hgcd(s, r_)
        return self.multiply_poly_matrices(M3, self.multiply_poly_matrices(M2, M1))
        
    def fast_extended_euclidean_algorithm(self, a, b):
        if self.degree(a) == self.degree(b):
            return self.fast_extended_euclidean_algorithm(b, self.poly_quotient_remainder(a, b)[1])
        elif self.degree(b) is None or self.degree(a) > self.degree(b):
            M = self.M_gcd(a, b)
            g = self.add_polys(self.mul_polys(M[0][0], a), self.mul_polys(M[0][1], b))
            v = self.inv(self.leading_coefficient(g))
            return self.mul_polys(self.truncate_poly(g), [v]), \
                   self.mul_polys(self.truncate_poly(M[0][1]), [v]), \
                   self.mul_polys(self.truncate_poly(M[0][0]), [v])
        else:
            return self.fast_extended_euclidean_algorithm(b, a)