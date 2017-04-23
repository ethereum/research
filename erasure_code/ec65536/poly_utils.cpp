#include "stdlib.h"
#include "stdio.h"
#include <iostream>
#include <vector>
#include <deque>

using namespace std;

vector<int> glogtable(65536, 0);
vector<int> gexptable(196608, 0);

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

vector<int> lagrange_interp(vector<int> ys, vector<int> xs) {
    int xs_size = xs.size();
    vector<int> root(xs_size + 1);
    root[xs_size] = 1;
    for (int i = 0; i < xs_size; i++) {
        int logx = glogtable[xs[i]];
        int offset = xs_size - i - 1;
        root[offset] = 0;
        for (int j = 0; j < i + 1; j++) {
            if (root[j + 1 + offset] and xs[i])
                root[j + offset] ^= gexptable[glogtable[root[j+1 + offset]] + logx];
        }
    }
    vector<int> b(xs_size);
    vector<int> output(root.size() - 1);
    for (int i = 0; i < xs_size; i++) {
        output[root.size() - 2] = 1;
        int logx = glogtable[xs[i]];
        for (int j = root.size() - 2; j > 0; j--) { 
            if (output[j] and xs[i])
                output[j - 1] = root[j] ^ gexptable[glogtable[output[j]] + logx];
            else
                output[j - 1] = root[j];
        }
        int denom = eval_poly_at(output, xs[i]);
        int log_yslice = glogtable[ys[i]] - glogtable[denom] + 65535;
        for (int j = 0; j < xs_size; j++) {
            if(output[j] and ys[i])
                b[j] ^= gexptable[glogtable[output[j]] + log_yslice];
        }
    }
    return b;
}


int main() {
    initialize_tables();
    //int myxs[] = {1, 2, 3, 4};
    //std::vector<int> xs (myxs, myxs + sizeof(myxs) / sizeof(int) );
    vector<int> xs(4096);
    vector<int> ys(4096);
    for (int v = 0; v < 4096; v++) {
        xs[v] = v;
        ys[v] = (v * v) % 65536;
    }
    vector<int> poly = lagrange_interp(ys, xs);
    unsigned int o = 0;
    for (int i = 4096; i < 8192; i++) {
        o += eval_poly_at(poly, i);
    }
    cout << o;
    //cout << eval_poly_at(poly, 0) << " " << ys[0] << "\n";
    //cout << eval_poly_at(poly, 134) << " " << ys[134] << "\n";
    //cout << eval_poly_at(poly, 375) << " " << ys[375] << "\n";
    //int o;
    //for (int i = 0; i < 524288; i ++)
    //    o += eval_poly_at(poly, i % 65536);
    //std::cout << o;
}
