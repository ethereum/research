#include <array>
#include <iostream>
#include <exception>
#include <cassert>
#include <cstdint>
#include <vector>

#include "utils.h"

class ZeroDivisionError : std::domain_error {
public:
    ZeroDivisionError() : domain_error("division by zero") { }
};

// GF(2^8) in the form (Z/2Z)[x]/(x^8+x^4+x^3+x+1)
// (the AES polynomial)
class Galois {
    // the coefficients of the polynomial, where the ith bit of `val` is the x^i
    // coefficient
    std::uint8_t v;

    // precomputed data: log and exp tables
    static const std::array<Galois, 255> exptable;
    static const std::array<std::uint8_t, 256> logtable;

public:
    explicit constexpr Galois(unsigned char val) : v(val) { }

    Galois operator+(Galois b) const {
        return Galois(v ^ b.v);
    }
    Galois operator-(Galois b) const {
        return Galois(v ^ b.v);
    }
    Galois operator*(Galois b) const {
        return v == 0 || b.v == 0
             ? Galois(0)
             : exptable[(unsigned(logtable[v]) + logtable[b.v]) % 255];
    }
    Galois operator/(Galois b) const {
        if (b.v == 0) {
            throw ZeroDivisionError();
        }
        return v == 0 || b.v == 0
             ? Galois(0)
             : exptable[(unsigned(logtable[v]) + 255u - logtable[b.v]) % 255];
    }
    Galois operator-() const {
        return *this;
    }

    Galois& operator+=(Galois b) {
        return *this = *this + b;
    }
    Galois& operator-=(Galois b) {
        return *this = *this - b;
    }
    Galois& operator*=(Galois b) {
        return *this = *this * b;
    }
    Galois& operator/=(Galois b) {
        return *this = *this / b;
    }

    bool operator==(Galois b) {
        return v == b.v;
    }

    // back door
    std::uint8_t val() const {
        return v;
    }
};

// Z/pZ, for an odd prime p
template<unsigned p>
class Modulo {
    // check that p is prime by trial division
    static constexpr bool is_prime(unsigned x, unsigned divisor = 2) {
        return divisor * divisor > x
               ? true
               : x % divisor != 0 && is_prime(x, divisor + 1);
    }
    static_assert(p > 2 && is_prime(p, 2), "p must be an odd prime!");

    unsigned v;

public:
    explicit Modulo(unsigned val) : v(val) {
        assert(v >= 0 && v < p);
    }


    Modulo inv() const {
        if (v == 0) {
            throw ZeroDivisionError();
        }
        unsigned r = 1, base = v, exp = p-2;
        while (exp > 0) {
            if (exp & 1) r = (r * base) % p;
            base = (base * base) % p;
            exp >>= 1;
        }
        return Modulo(r);
    }
    Modulo operator+(Modulo b) const {
        return Modulo((v + b.v) % p);
    }
    Modulo operator-(Modulo b) const {
        return Modulo((v + p - b.v) % p);
    }
    Modulo operator*(Modulo b) const {
        return Modulo((v * b.v) % p);
    }
    Modulo operator/(Modulo b) const {
        return *this * b.inv();
    }

    Modulo& operator+=(Modulo b) {
        return *this = *this + b;
    }
    Modulo& operator-=(Modulo b) {
        return *this = *this - b;
    }
    Modulo& operator*=(Modulo b) {
        return *this = *this * b;
    }
    Modulo& operator/=(Modulo b) {
        return *this = *this / b;
    }

    bool operator==(Modulo b) {
        return v == b.v;
    }

    // back door
    unsigned val() const {
        return v;
    }
};

// Evaluates a polynomial p in little-endian form (e.g. x^2 + 3x + 2 is
// represented as {2, 3, 1}) at coordinate x,
// e.g. eval_poly_at((int[]){2, 3, 1}, 5) = 42.
//
// T should be a type supporting ring arithmetic and T(0) and T(1) should be the
// appropriate identities.
//
// Range should be a type that can be iterated to get const T& elements.
template<typename T, typename Range>
T eval_poly_at(const Range& p, T x) {
    T r(0), xi(1);
    for (const T& c_i : p) {
        r += c_i * xi;
        xi *= x;
    }
    return r;
}

// Given p+1 y values and x values with no errors, recovers the original
// degree-p polynomial. For example,
// lagrange_interp<double>((double[]){51.0, 59.0, 66.0},
//                         (double[]){1.0, 3.0, 4.0})
// = {50.0, 0.0, 1.0}.
//
// T should be a field and Range should be a sized range type with values of
// type T.  T(0) and T(1) should be the appropriate field identities.
template<typename T, typename Range>
std::vector<T> lagrange_interp(const Range& pieces, const Range& xs) {
    // `size` is the number of datapoints; the degree of the result polynomial
    // is then `size-1`
    const unsigned size = pieces.size();
    assert(size == xs.size());

    std::vector<T> root{T(1)}; // initially just the polynomial "1"
    // build up the numerator polynomial, `root`, by taking the product of (x-v)
    // (implemented as convolving repeatedly with [-v, 1])
    for (const T& v : xs) {
        // iterate backward since new root[i] depends on old root[i-1]
        for (unsigned i = root.size(); i--; ) {
            root[i] *= -v;
            if (i > 0) root[i] += root[i-1];
        }
        // polynomial is always monic so save an extra multiply by doing this
        // after
        root.emplace_back(1);
    }
    // should have degree `size`
    assert(root.size() == size + 1);

    // generate per-value numerator polynomials by dividing the master
    // polynomial back by each x coordinate
    std::vector<std::vector<T> > nums;
    nums.reserve(size);
    for (const T& v : xs) {
        // divide `root` by (x-v) to get a degree size-1 polynomial
        // (i.e. with `size` coefficients)
        std::vector<T> num(size, T(0));
        // compute the x^0, x^1, ..., x^(p-2) coefficients by long division
        T last = num.back() = T(1); // still always a monic polynomial
        for (int i = int(size)-2; i >= 0; --i) {
            num[i] = last = root[i+1] + last * v;
        }
        nums.emplace_back(std::move(num));
    }
    assert(nums.size() == size);

    // generate denominators by evaluating numerator polys at their x
    std::vector<T> denoms;
    denoms.reserve(size);
    {
        unsigned i = 0;
        for (const T& v : xs) {
            denoms.push_back(eval_poly_at(nums[i], v));
            ++i;
        }
    }
    assert(denoms.size() == size);

    // generate output polynomial by taking the sum over i of
    // (nums[i] * pieces[i] / denoms[i])
    std::vector<T> sum(size, T(0));
    {
        unsigned i = 0;
        for (const T& y : pieces) {
            T factor = y / denoms[i];
            // add nums[i] * factor to sum, as a vector
            for (unsigned j = 0; j < size; ++j) {
                sum[j] += nums[i][j] * factor;
            }
            ++i;
        }
    }
    return sum;
}

// Given two linear equations, eliminates the first variable and returns
// the resulting equation.
//
// An equation of the form a_1 x_1 + ... + a_n x_n + b = 0
// is represented as the array [a_1, ..., a_n, b].
//
// T should be a ring and Range should be an indexable, sized range of T.
template<typename T, typename Range>
std::vector<T> elim(const Range& a, const Range& b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    const unsigned size = a.size();
    for (unsigned i = 1; i < size; ++i) {
        result.push_back(a[i] * b[0] - b[i] * a[0]);
    }
    return result;
}

// Given one homogeneous linear equation and the values of all but the first
// variable, solve for the value of the first variable.
//
// For an equation of the form
//     a_1 x_1 + ... + a_n x_n = 0
// pass two arrays, [a_1, ..., a_n] and [x_2, ..., x_n].
//
// T should be a field; and R1 and R2 should be indexable, sized ranges of T.
template<typename T, typename R1, typename R2>
T evaluate(const R1& coeffs, const R2& vals) {
    assert(coeffs.size() == vals.size() + 1);
    T total(0);
    const unsigned size = vals.size();
    for (unsigned i = 0; i < size; ++i) {
        total -= coeffs[i+1] * vals[i];
    }
    return total / coeffs[0];
}

// Given an n*n system of inhomogeneous linear equations, solve for the value of
// every variable.
//
// For equations of the form
//     a_1,1 x_1 + ... + a_1,n x_n + b_1 = 0
//     a_2,1 x_1 + ... + a_2,n x_n + b_2 = 0
//     ...
//     a_n,1 x_1 + ... + a_n,n x_n + b_n = 0
// pass a two-dimensional array
//     [[a_1,1, ..., a_1,n, b_1], ..., [a_n,1, ..., a_n,n, b_n]].
//
// Returns the values of [x_1, ..., x_n].
//
// T should be a field.
template<typename T>
std::vector<T> sys_solve(std::vector<std::vector<T>> eqs) {
    assert(eqs.size() > 0);
    std::vector<std::vector<T>> back_eqs{eqs[0]};

    while (eqs.size() > 1) {
        std::vector<std::vector<T>> neweqs;
        neweqs.reserve(eqs.size()-1);
        for (unsigned i = 0; i < eqs.size()-1; ++i) {
            neweqs.push_back(elim<T>(eqs[i], eqs[i+1]));
        }
        eqs = std::move(neweqs);
        // find a row with a nonzero first entry
        unsigned i = 0;
        while (i + 1 < eqs.size() && eqs[i][0] == T(0)) {
            ++i;
        }
        back_eqs.push_back(eqs[i]);
    }

    std::vector<T> kvals(back_eqs.size()+1, T(0));
    kvals.back() = T(1);
    // back-substitute in reverse order
    // (smallest to largest equation)
    for (unsigned i = back_eqs.size(); i--; ) {
        kvals[i] = evaluate<T>(back_eqs[i],
                // use the already-computed values + the 1 at the end
                make_iter_pair(kvals.begin()+i+1, kvals.end()));
    }

    kvals.pop_back();

    return kvals;
}

// Divide two polynomials with nonzero leading terms.
// T should be a field.
template<typename T>
std::vector<T> polydiv(std::vector<T> Q, const std::vector<T>& E) {
    if (Q.size() < E.size()) return {};
    std::vector<T> div(Q.size() - E.size() + 1, T(0));
    unsigned i = div.size();
    while (i--) {
        T factor = Q.back() / E.back();
        div[i] = factor;
        // subtract factor * E * x^i from Q
        Q.pop_back(); // the highest term should cancel
        for (unsigned j = 0; j < E.size() - 1; ++j) {
            Q[i+j] -= factor * E[j];
        }
        assert(Q.size() == i + E.size() - 1);
    }
    return div;
}

// Given a set of y coordinates and x coordinates, and the degree of the
// original polynomial, determines the original polynomial even if some of the y
// coordinates are wrong. If m is the minimal number of pieces (ie.  degree +
// 1), t is the total number of pieces provided, then the algo can handle up to
// (t-m)/2 errors.
//
// T should be a field. In particular, division by zero over T should throw
// ZeroDivisionError.
template<typename T>
std::vector<T> berlekamp_welch_attempt(const std::vector<T>& pieces,
        const std::vector<T>& xs, unsigned master_degree) {
    const unsigned error_locator_degree = (pieces.size() - master_degree - 1) / 2;
    // Set up the equations for y[i]E(x[i]) = Q(x[i])
    // degree(E) = error_locator_degree
    // degree(Q) = master_degree + error_locator_degree - 1
    std::vector<std::vector<T>> eqs(2*error_locator_degree + master_degree + 1);
    for (unsigned i = 0; i < eqs.size(); ++i) {
        std::vector<T>& eq = eqs[i];
        const T& x = xs[i];
        const T& piece = pieces[i];
        T neg_x_j = T(0) - T(1);
        for (unsigned j = 0; j < error_locator_degree + master_degree + 1; ++j) {
            eq.push_back(neg_x_j);
            neg_x_j *= x;
        }
        T x_j = T(1);
        for (unsigned j = 0; j < error_locator_degree + 1; ++j) {
            eq.push_back(x_j * piece);
            x_j *= x;
        }
    }
    // Solve the equations
    // Assume the top error polynomial term to be one
    int errors = error_locator_degree;
    unsigned ones = 1;
    std::vector<T> polys;
    while (errors >= 0) {
        try {
            polys = sys_solve(eqs);
        } catch (const ZeroDivisionError&) {
            eqs.pop_back();
            for (auto& eq : eqs) {
                eq[eq.size()-2] += eq.back();
                eq.pop_back();
            }
            --errors;
            ++ones;
            continue;
        }
        for (unsigned i = 0; i < ones; ++i) polys.emplace_back(1);
        break;
    }
    if (errors < 0) {
        throw std::logic_error("Not enough data!");
    }
    // divide the polynomials...
    const unsigned split = error_locator_degree + master_degree + 1;
    std::vector<T> div = polydiv(std::vector<T>(polys.begin(), polys.begin() + split),
                                 std::vector<T>(polys.begin() + split, polys.end()));
    unsigned corrects = 0;
    for (unsigned i = 0; i < xs.size(); ++i) {
        if (eval_poly_at<T>(div, xs[i]) == pieces[i]) {
            ++corrects;
        }
    }
    if (corrects < master_degree + errors) {
        throw std::logic_error("Answer doesn't match (too many errors)!");
    }
    return div;
}

// Extends a list of integers in [0 ... 255] (if using Galois arithmetic) by
// adding n redundant error-correction values
template<typename T, typename F=Galois>
std::vector<T> extend(std::vector<T> data, unsigned n) {
    const unsigned size = data.size();

    std::vector<F> data_f;
    data_f.reserve(size);
    for (T d : data) data_f.emplace_back(d);

    std::vector<F> xs;
    for (unsigned i = 0; i < size; ++i) xs.emplace_back(i);

    std::vector<F> poly = berlekamp_welch_attempt(data_f, xs, size-1);

    data.reserve(size+n);
    for (unsigned i = 0; i < n; ++i) {
        data.push_back(eval_poly_at(poly, F(T(size + i))).val());
    }
    return data;
}

// Repairs a list of integers in [0 ... 255]. Some integers can be erroneous,
// and you can put -1 in place of an integer if you know that a certain
// value is defective or missing. Uses the Berlekamp-Welch algorithm to
// do error-correction
template<typename T, typename F=Galois>
std::vector<T> repair(const std::vector<T>& data, unsigned datasize) {
    std::vector<F> vs, xs;
    for (unsigned i = 0; i < data.size(); ++i) {
        if (data[i] >= 0) {
            vs.emplace_back(data[i]);
            xs.emplace_back(T(i));
        }
    }
    std::vector<F> poly = berlekamp_welch_attempt(vs, xs, datasize - 1);
    std::vector<T> result;
    for (unsigned i = 0; i < data.size(); ++i) {
        result.push_back(eval_poly_at(poly, F(T(i))).val());
    }
    return result;
}


template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& d) {
    assert(d.size() > 0);
    unsigned width = d[0].size();
    std::vector<std::vector<T>> result(width);
    for (unsigned i = 0; i < width; ++i) {
        for (unsigned j = 0; j < d.size(); ++j) {
            result[i].push_back(d[j][i]);
        }
    }
    return result;
}

template<typename T>
std::vector<T> extract_column(const std::vector<std::vector<T>>& d, unsigned i) {
    std::vector<T> result;
    for (unsigned j = 0; j < d.size(); ++j) {
        result.push_back(d[j][i]);
    }
    return result;
}

// Extends a list of bytearrays
// eg. extend_chunks([map(ord, 'hello'), map(ord, 'world')], 2)
// n is the number of redundant error-correction chunks to add
template<typename T, typename F=Galois>
std::vector<std::vector<T>> extend_chunks(
        const std::vector<std::vector<T>>& data,
        unsigned n) {
    std::vector<std::vector<T>> o;
    const unsigned height = data.size();
    assert(height > 0);
    const unsigned width = data[0].size();
    for (unsigned i = 0; i < width; ++i) {
        o.push_back(extend<T, F>(extract_column(data, i), n));
    }
    return transpose(o);
}

// Repairs a list of bytearrays. Use an empty array in place of a missing array.
// Individual arrays can contain some missing or erroneous data.
template<typename T, typename F=Galois>
std::vector<std::vector<T>> repair_chunks(
        std::vector<std::vector<T>> data,
        unsigned datasize) {
    unsigned width = 0;
    for (const std::vector<T>& row : data) {
        if (row.size() > 0) {
            width = row.size();
            break;
        }
    }
    assert(width > 0);
    for (std::vector<T>& row : data) {
        if (row.size() == 0) {
            row.assign(width, -1);
        } else {
            assert(row.size() == width);
        }
    }
    std::vector<std::vector<T>> o;
    for (unsigned i = 0; i < width; ++i) {
        o.push_back(repair<T, F>(extract_column(data, i), datasize));
    }
    return transpose(o);
}

// Extends either a bytearray or a list of bytearrays or a list of lists...
// Used in the cubify method to expand a cube in all dimensions
template<typename T, typename F=Galois>
struct deep_extend_chunks_helper {
    static std::vector<T> go(const std::vector<T>& data, unsigned n) {
        return extend<T, Galois>(data, n);
    }
};
template<typename T, typename F>
struct deep_extend_chunks_helper<std::vector<T>, F> {
    static std::vector<std::vector<T>> go(const std::vector<std::vector<T>>& data, unsigned n) {
        std::vector<std::vector<T>> o;
        const unsigned height = data.size();
        assert(height > 0);
        const unsigned width = data[0].size();
        for (unsigned i = 0; i < width; ++i) {
            o.push_back(deep_extend_chunks_helper<T, F>::go(extract_column(data, i), n));
        }
        return transpose(o);
    }
};
template<typename T, typename F=Galois>
std::vector<T> deep_extend_chunks(const std::vector<T>& data, unsigned n) {
    return deep_extend_chunks_helper<T, F>::go(data, n);
}
