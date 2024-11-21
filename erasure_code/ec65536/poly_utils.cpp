#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <vector>
#include <deque>

using namespace std;

vector<int> glogtable(65536, 0);
vector<int> gexptable(196608, 0);

const int ROOT_CUTOFF = 32;

void initialize_tables() {
    int v = 1;
    for (int i = 0; i < 65536; i++) {
        glogtable[v] = i;
        gexptable[i] = v;
        gexptable[i + 65535] = v;
        gexptable[i + 131070] = v;
        if (v & 32768)
            v = (v * 2) ^ v ^ 103425;
        else
            v = (v * 2) ^ v;
    }
}

int eval_poly_at(vector<int> poly, int x) {
    if (x == 0)
        return poly[0];
    int logx = glogtable[x];
    int y = 0;
    for (int i = 0; i < poly.size(); i++) {
        if (poly[i])
            y ^= gexptable[(logx * i + glogtable[poly[i]]) % 65535];
    }
    return y;
}

int eval_log_poly_at(vector<int> poly, int x) {
    if (x == 0)
        return poly[0] == 65537 ? 0 : gexptable[poly[0]];
    int logx = glogtable[x];
    int y = 0;
    for (int i = 0; i < poly.size(); i++) {
        if (poly[i] != 65537)
            y ^= gexptable[(logx * i + poly[i]) % 65535];
    }
    return y;
}

// Compute the product of two (equal length) polynomials. Takes ~O(N ** 1.59) time.
vector<int> karatsuba_mul(vector<int> p, vector<int> q) {
    int L = p.size();
    if (L <= 64) {
        vector<int> o(L * 2);
        vector<int> logq(L);
        for (int i = 0; i < L; i++) logq[i] = glogtable[q[i]];
        for (int i = 0; i < L; i++) {
            int log_pi = glogtable[p[i]];
            for (int j = 0; j < L; j++) {
                if (p[i] && q[j])
                    o[i + j] ^= gexptable[log_pi + logq[j]];
            }
        }
        return o;
    }
    if (L % 2) {
        L += 1;
        p.push_back(0);
        q.push_back(0);
    }
    int halflen = L / 2;
    vector<int> low1 = vector<int>(p.begin(), p.begin() + halflen);
    vector<int> low2 = vector<int>(q.begin(), q.begin() + halflen);
    vector<int> high1 = vector<int>(p.begin() + halflen, p.end());
    vector<int> high2 = vector<int>(q.begin() + halflen, q.end());
    vector<int> sum1(halflen);
    vector<int> sum2(halflen);
    for (int i = 0; i < halflen; i++) {
        sum1[i] = low1[i] ^ high1[i];
        sum2[i] = low2[i] ^ high2[i];
    }
    vector<int> z0 = karatsuba_mul(low1, low2);
    vector<int> z2 = karatsuba_mul(high1, high2);
    vector<int> m = karatsuba_mul(sum1, sum2);
    vector<int> o(L * 2);
    for (int i = 0; i < L; i++) {
        o[i] ^= z0[i];
        o[i + halflen] ^= (m[i] ^ z0[i] ^ z2[i]);
        o[i + L] ^= z2[i];
    }
    return o;
}

vector<int> mk_root(vector<int> xs) {
    int L = xs.size();
    if (L >= ROOT_CUTOFF) {
        int halflen = L / 2;
        vector<int> left = vector<int>(xs.begin(), xs.begin() + halflen);
        vector<int> right = vector<int>(xs.begin() + halflen, xs.end());
        vector<int> o = karatsuba_mul(mk_root(left), mk_root(right));
        o.resize(L + 1);
        return o;
    }
    vector<int> root(L + 1);
    root[L] = 1;
    for (int i = 0; i < L; i++) {
        int logx = glogtable[xs[i]];
        int offset = L - i - 1;
        root[offset] = 0;
        for (int j = offset; j < i + 1 + offset; j++) {
            if (root[j + 1] and xs[i])
                root[j] ^= gexptable[glogtable[root[j+1]] + logx];
        }
    }
    return root;
}

vector<int> subroot_linear_combination(vector<int> xs, vector<int> factors) {
    int L = xs.size();
    /*if (L <= ROOT_CUTOFF) {
        vector<int> out(L + 1);
        vector<int> root = mk_root(xs);
        for (int i = 0; i < L; i++) {
            vector<int> output(L + 1);
            output[root.size() - 2] = 1;
            int logx = glogtable[xs[i]];
            if (factors[i]) {
                int log_fac = glogtable[factors[i]];
                for (int j = root.size() - 2; j > 0; j--) { 
                    if (output[j] and xs[i])
                        output[j - 1] = root[j] ^ gexptable[glogtable[output[j]] + logx];
                    else
                        output[j - 1] = root[j];
                    out[j] ^= gexptable[glogtable[output[j]] + log_fac];
                }
                out[0] ^= gexptable[glogtable[output[0]] + log_fac];
            }
        }
        return out;
    }*/
    if (L == 1) {
        vector<int> o(2);
        o[0] = factors[0];
        return o;
    }
    int halflen = L / 2;
    vector<int> xs_left = vector<int>(xs.begin(), xs.begin() + halflen);
    vector<int> xs_right = vector<int>(xs.begin() + halflen, xs.end());
    vector<int> factors_left = vector<int>(factors.begin(), factors.begin() + halflen);
    vector<int> factors_right = vector<int>(factors.begin() + halflen, factors.end());
    vector<int> R1 = mk_root(xs_left);
    vector<int> R2 = mk_root(xs_right);
    vector<int> o1 = karatsuba_mul(R1, subroot_linear_combination(xs_right, factors_right));
    vector<int> o2 = karatsuba_mul(R2, subroot_linear_combination(xs_left, factors_left));
    vector<int> o(L + 1);
    for (int i = 0; i < L; i++) {
        o[i] = o1[i] ^ o2[i];
    }
    return o;
}


vector<int> derivative_and_square_base(vector<int> p) {
    vector<int> o((p.size() - 1) / 2);
    for (int i = 0; i < o.size(); i+= 1) {
        o[i] = p[i * 2 + 1];
    }
    return o;
}

vector<int> poly_to_logs(vector<int> p) {
    vector<int> o(p.size());
    for (int i = 0; i < p.size(); i++) {
        if (p[i])
            o[i] = glogtable[p[i]];
        else
            o[i] = 65537;
    }
    return o;
}

vector<int> lagrange_interp(vector<int> ys, vector<int> xs) {
    int xs_size = xs.size();
    vector<int> root = mk_root(xs);
    vector<int> log_rootprime = poly_to_logs(derivative_and_square_base(root));
    vector<int> factors(xs_size);
    for (int i = 0; i < xs_size; i++) {
        int x_square = xs[i] ? gexptable[glogtable[xs[i]] * 2] : 0;
        int denom = eval_log_poly_at(log_rootprime, x_square);
        if (ys[i])
            factors[i] = gexptable[glogtable[ys[i]] + 65535 - glogtable[denom]];
    }
    return subroot_linear_combination(xs, factors);
}

const int SIZE = 4096;


int main() {
    initialize_tables();
    //int myxs[] = {1, 2, 3, 4};
    //std::vector<int> xs (myxs, myxs + sizeof(myxs) / sizeof(int) );
    vector<int> xs(SIZE);
    vector<int> ys(SIZE);
    for (int v = 0; v < SIZE; v++) {
        ys[v] = v * 3;
        xs[v] = 1000 + v * 7;
    }
    //vector<int> d = derivative(mk_root(xs));
    //for (int i = 0; i < d.size(); i++) cout << d[i] << " ";
    //cout << "\n";
    /*vector<int> prod = mk_root(xs);
    vector<int> prod = karatsuba_mul(xs, ys);
    for (int i = 0; i < SIZE + 1; i++)
        cout << prod[i] << " ";
    cout << "\n";
    cout << eval_poly_at(prod, 189) << " " << gexptable[glogtable[eval_poly_at(xs, 189)] + glogtable[eval_poly_at(ys, 189)]] << "\n";*/
    for (int a = 0; a < 10; a++) {
        ys[0] = a;
        vector<int> poly = lagrange_interp(ys, xs);
        vector<int> logpoly = poly_to_logs(poly);
        cout << eval_poly_at(poly, 1700) << "\n";
        unsigned int o = 0;
        for (int i = SIZE; i < SIZE * 2; i++) {
            o += eval_log_poly_at(logpoly, i);
        }
        cout << o << "\n";
    }
    //cout << eval_poly_at(poly, 0) << " " << ys[0] << "\n";
    //cout << eval_poly_at(poly, 134) << " " << ys[134] << "\n";
    //cout << eval_poly_at(poly, 375) << " " << ys[375] << "\n";
    //int o;
    //for (int i = 0; i < 524288; i ++)
    //    o += eval_poly_at(poly, i % 65536);
    //std::cout << o;
}
