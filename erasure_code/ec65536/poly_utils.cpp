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
    deque<int> root;
    root.push_back(1);
    for (int i = 0; i < xs.size(); i++) {
        int logx = glogtable[xs[i]];
        root.push_front(0);
        for (int j = 0; j < root.size() - 1; j++) {
            if (root[j + 1] and xs[i])
                root[j] ^= gexptable[glogtable[root[j+1]] + logx];
        }
    }
    vector<vector<int> > nums;
    for (int i = 0; i < xs.size(); i++) {
        vector<int> output(root.size() - 1);
        output[root.size() - 2] = 1;
        int logx = glogtable[xs[i]];
        for (int j = root.size() - 2; j > 0; j--) { 
            if (output[j] and xs[i])
                output[j - 1] = root[j] ^ gexptable[glogtable[output[j]] + logx];
            else
                output[j - 1] = root[j];
        }
        nums.push_back(output);
    }
    vector<int> denoms;
    for (int i = 0; i < xs.size(); i++) {
        denoms.push_back(eval_poly_at(nums[i], xs[i]));
    }
    vector<int> b(xs.size());
    for (int i = 0; i < xs.size(); i++) {
        int log_yslice = glogtable[ys[i]] - glogtable[denoms[i]] + 65535;
        for (int j = 0; j < xs.size(); j++) {
            if(nums[i][j] and ys[i])
                b[j] ^= gexptable[glogtable[nums[i][j]] + log_yslice];
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
