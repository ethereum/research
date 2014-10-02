(function() {
    var me = {};

    function ZeroDivisionError() {
        if (!this) return new ZeroDivisionError();
        this.message = "division by zero";
        this.name = "ZeroDivisionError";
    }
    me.ZeroDivisionError = ZeroDivisionError;

    // per-byte 2^8 Galois field
    // Note that this imposes a hard limit that the number of extended chunks can
    // be at most 256 along each dimension
    function galoistpl(a) {
        // 2 is not a primitive root, so we have to use 3 as our logarithm base
        var r = a ^ (a<<1); // a * (x+1)
        if (r > 0xff) { // overflow?
            r = r ^ 0x11b;
        }
        return r;
    }

    // Precomputing a multiplication and XOR table for increased speed
    var glogtable = new Array(256);
    var gexptable = [];
    (function() {
        var v = 1;
        for (var i = 0; i < 255; i++) {
            glogtable[v] = i;
            gexptable.push(v);
            v = galoistpl(v);
        }
    })();
    me.glogtable = glogtable;
    me.gexptable = gexptable;

    function Galois(val) {
        if (!(this instanceof Galois)) return new Galois(val);
        if (val instanceof Galois) {
            this.val = val.val;
        } else {
            this.val = val;
        }
        if (typeof Object.freeze == 'function') {
            Object.freeze(this);
        }
    }
    me.Galois = Galois;
    Galois.prototype.add = Galois.prototype.sub = function(other) {
        return new Galois(this.val ^ other.val);
    };
    Galois.prototype.mul = function(other) {
        if (this.val == 0 || other.val == 0) {
            return new Galois(0);
        }
        return new Galois(gexptable[(glogtable[this.val] +
                    glogtable[other.val]) % 255]);
    };
    Galois.prototype.div = function(other) {
        if (other.val == 0) {
            throw new ZeroDivisionError();
        }
        if (this.val == 0) {
            return new Galois(0);
        }
        return new Galois(gexptable[(glogtable[this.val] + 255 -
                    glogtable[other.val]) % 255]);
    };
    Galois.prototype.inspect = function() {
        return ""+this.val;
    };

    function powmod(b, e, m) {
        var r = 1;
        while (e > 0) {
            if (e & 1) r = (r * b) % m;
            b = (b * b) % m;
            e = e >> 1;
        }
        return r;
    }


    // Modular division class
    function mkModuloClass(n) {
        if (n <= 2) throw new Error("n must be prime!");
        for (var divisor = 2; divisor * divisor <= n; divisor++) {
            if (n % divisor == 0) {
                throw new Error("n must be prime!");
            }
        }

        function Mod(val) {
            if (!(this instanceof Mod)) return new Mod(val);
            if (val instanceof Mod) {
                this.val = val.val;
            } else {
                this.val = val;
            }
            if (typeof Object.freeze == 'function') {
                Object.freeze(this);
            }
        }
        Mod.modulo = n;
        Mod.prototype.add = function(other) {
            return new Mod((this.val + other.val) % n);
        };
        Mod.prototype.sub = function(other) {
            return new Mod((this.val + n - other.val) % n);
        };
        Mod.prototype.mul = function(other) {
            return new Mod((this.val * other.val) % n);
        };
        Mod.prototype.div = function(other) {
            return new Mod((this.val * powmod(other.val, n-2, n)) % n);
        };
        Mod.prototype.inspect = function() {
            return ""+this.val;
        };

        return Mod;
    }
    me.mkModuloClass = mkModuloClass;

    // Evaluates a polynomial in little-endian form, eg. x^2 + 3x + 2 = [2, 3, 1]
    // (normally I hate little-endian, but in this case dealing with polynomials
    // it's justified, since you get the nice property that p[n] is the nth degree
    // term of p) at coordinate x, eg. eval_poly_at([2, 3, 1], 5) = 42 if you are
    // using float as your arithmetic
    function eval_poly_at(p, x) {
        var arithmetic = p[0].constructor;
        var y = new arithmetic(0);
        var x_to_the_i = new arithmetic(1);
        for (var i = 0; i < p.length; i++) {
            y = y.add(x_to_the_i.mul(p[i]))
                x_to_the_i = x_to_the_i.mul(x);
        }
        return y;
    }
    me.eval_poly_at = eval_poly_at;

    // Given p+1 y values and x values with no errors, recovers the original
    // p+1 degree polynomial. For example,
    // lagrange_interp([51.0, 59.0, 66.0], [1, 3, 4]) = [50.0, 0, 1.0]
    // if you are using float as your arithmetic
    function lagrange_interp(pieces, xs) {
        var arithmetic = pieces[0].constructor;
        var zero = new arithmetic(0);
        var one = new arithmetic(1);
        // Generate master numerator polynomial
        var root = [one];
        var i, j;
        for (i = 0; i < xs.length; i++) {
            root.unshift(zero);
            for (j = 0; j < root.length - 1; j++) {
                root[j] = root[j].sub(root[j+1].mul(xs[i]));
            }
        }
        // Generate per-value numerator polynomials by dividing the master
        // polynomial back by each x coordinate
        var nums = [];
        for (i = 0; i < xs.length; i++) {
            var output = [];
            var last = one;
            for (j = 2; j < root.length+1; j++) {
                output.unshift(last);
                if (j != root.length) {
                    last = root[root.length-j].add(last.mul(xs[i]));
                }
            }
            nums.push(output);
        }
        // Generate denominators by evaluating numerator polys at their x
        var denoms = [];
        for (i = 0; i < xs.length; i++) {
            var denom = zero;
            var x_to_the_j = one;
            for (j = 0; j < nums[i].length; j++) {
                denom = denom.add(x_to_the_j.mul(nums[i][j]));
                x_to_the_j = x_to_the_j.mul(xs[i]);
            }
            denoms.push(denom);
        }
        // Generate output polynomial
        var b = [];
        for (i = 0; i < pieces.length; i++) {
            b[i] = zero;
        }
        for (i = 0; i < xs.length; i++) {
            var yslice = pieces[i].div(denoms[i]);
            for (j = 0; j < pieces.length; j++) {
                b[j] = b[j].add(nums[i][j].mul(yslice));
            }
        }
        return b;
    }
    me.lagrange_interp = lagrange_interp;

    // Compresses two linear equations of length n into one
    // equation of length n-1
    // Format:
    // 3x + 4y = 80 (ie. 3x + 4y - 80 = 0) -> a = [3,4,-80]
    // 5x + 2y = 70 (ie. 5x + 2y - 70 = 0) -> b = [5,2,-70]
    function elim(a, b) {
        var c = [];
        for (var i = 1; i < a.length; i++) {
            c[i-1] = a[i].mul(b[0]).sub(b[i].mul(a[0]));
        }
        return c;
    }

    // Linear equation solver
    // Format:
    // 3x + 4y = 80, y = 5 (ie. 3x + 4y - 80z = 0, y = 5, z = 1)
    //      -> coeffs = [3,4,-80], vals = [5,1]
    function evaluate(coeffs, vals) {
        var arithmetic = coeffs[0].constructor;
        var tot = new arithmetic(0);
        for (var i = 0; i < vals.length; i++) {
            tot = tot.sub(coeffs[i+1].mul(vals[i]));
        }
        if (coeffs[0].val == 0) {
            throw new ZeroDivisionError();
        }
        return tot.div(coeffs[0]);
    }

    // Linear equation system solver
    // Format:
    // ax + by + c = 0, dx + ey + f = 0
    // -> [[a, b, c], [d, e, f]]
    // eg.
    // [[3.0, 5.0, -13.0], [9.0, 1.0, -11.0]] -> [1.0, 2.0]
    function sys_solve(eqs) {
        var arithmetic = eqs[0][0].constructor;
        var one = new arithmetic(1);
        var back_eqs = [eqs[0]];
        var i;
        while (eqs.length > 1) {
            var neweqs = [];
            for (i = 0; i < eqs.length - 1; i++) {
                neweqs.push(elim(eqs[i], eqs[i+1]));
            }
            eqs = neweqs;
            i = 0;
            while (i < eqs.length - 1 && eqs[i][0].val == 0) {
                i++;
            }
            back_eqs.unshift(eqs[i]);
        }
        var kvals = [one];
        for (i = 0; i < back_eqs.length; i++) {
            kvals.unshift(evaluate(back_eqs[i], kvals));
        }
        return kvals.slice(0, -1);
    }
    me.sys_solve = sys_solve;

    function polydiv(Q, E) {
        var qpoly = Q.slice();
        var epoly = E.slice();
        var div = [];
        while (qpoly.length >= epoly.length) {
            div.unshift(qpoly[qpoly.length-1].div(epoly[epoly.length-1]));
            for (var i = 2; i < epoly.length + 1; i++) {
                qpoly[qpoly.length-i] =
                    qpoly[qpoly.length-i].sub(div[0].mul(epoly[epoly.length-i]));
            }
            qpoly.pop();
        }
        return div;
    }
    me.polydiv = polydiv;

    // Given a set of y coordinates and x coordinates, and the degree of the
    // original polynomial, determines the original polynomial even if some of
    // the y coordinates are wrong. If m is the minimal number of pieces (ie.
    // degree + 1), t is the total number of pieces provided, then the algo can
    // handle up to (t-m)/2 errors. See:
    // http://en.wikipedia.org/wiki/Berlekamp%E2%80%93Welch_algorithm#Example
    // (just skip to my example, the rest of the article sucks imo)
    function berlekamp_welch_attempt(pieces, xs, master_degree) {
        var error_locator_degree = Math.floor((pieces.length - master_degree - 1) / 2);
        var arithmetic = pieces[0].constructor;
        var zero = new arithmetic(0);
        var one = new arithmetic(1);
        // Set up the equations for y[i]E(x[i]) = Q(x[i])
        // degree(E) = error_locator_degree
        // degree(Q) = master_degree + error_locator_degree - 1
        var eqs = [];
        var i,j;
        for (i = 0; i < 2 * error_locator_degree + master_degree + 1; i++) {
            eqs.push([]);
        }
        for (i = 0; i < 2 * error_locator_degree + master_degree + 1; i++) {
            var neg_x_to_the_j = zero.sub(one);
            for (j = 0; j < error_locator_degree + master_degree + 1; j++) {
                eqs[i].push(neg_x_to_the_j);
                neg_x_to_the_j = neg_x_to_the_j.mul(xs[i]);
            }
            var x_to_the_j = one;
            for (j = 0; j < error_locator_degree + 1; j++) {
                eqs[i].push(x_to_the_j.mul(pieces[i]));
                x_to_the_j = x_to_the_j.mul(xs[i]);
            }
        }
        // Solve 'em
        // Assume the top error polynomial term to be one
        var errors = error_locator_degree;
        var ones = 1;
        var polys;
        while (errors >= 0) {
            try {
                polys = sys_solve(eqs);
                for (i = 0; i < ones; i++) polys.push(one);
                break;
            } catch (e) {
                if (e instanceof ZeroDivisionError) {
                    eqs.pop();
                    for (i = 0; i < eqs.length; i++) {
                        var eq = eqs[i];
                        eq[eq.length-2] = eq[eq.length-2].add(eq[eq.length-1]);
                        eq.pop();
                    }
                    errors--;
                    ones++;
                } else {
                    throw e;
                }
            }
        }
        if (errors < 0) {
            throw new Error("Not enough data!");
        }
        // Divide the polynomials
        var qpoly = polys.slice(0, error_locator_degree + master_degree + 1);
        var epoly = polys.slice(error_locator_degree + master_degree + 1);
        var div = polydiv(qpoly, epoly);
        // Check
        var corrects = 0;
        for (i = 0; i < xs.length; i++) {
            if (eval_poly_at(div, xs[i]).val == pieces[i].val) {
                corrects++;
            }
        }
        if (corrects < master_degree + errors) {
            throw new Error("Answer doesn't match (too many errors)!");
        }
        return div;
    }
    me.berlekamp_welch_attempt = berlekamp_welch_attempt;

    // Extends a list of integers in [0 ... 255] (if using Galois arithmetic) by
    // adding n redundant error-correction values
    function extend(data, n, arithmetic) {
        arithmetic = arithmetic || Galois;
        function mk(x) { return new arithmetic(x); }
        var data2 = data.map(mk);
        var data3 = data.slice();
        var xs = [];
        var i;
        for (i = 0; i < data.length; i++) {
            xs.push(new arithmetic(i));
        }
        var poly = berlekamp_welch_attempt(data2, xs, data.length - 1);
        for (i = 0; i < n; i++) {
            data3.push(eval_poly_at(poly, new arithmetic(data.length + i)).val);
        }
        return data3;
    }
    me.extend = extend;

    // Repairs a list of integers in [0 ... 255]. Some integers can be
    // erroneous, and you can put null (or undefined) in place of an integer if
    // you know that a certain value is defective or missing. Uses the
    // Berlekamp-Welch algorithm to do error-correction
    function repair(data, datasize, arithmetic) {
        arithmetic = arithmetic || Galois;
        var vs = [];
        var xs = [];
        var i;
        for (var i = 0; i < data.length; i++) {
            if (data[i] != null) {
                vs.push(new arithmetic(data[i]));
                xs.push(new arithmetic(i));
            }
        }
        var poly = berlekamp_welch_attempt(vs, xs, datasize - 1);
        var result = [];
        for (i = 0; i < data.length; i++) {
            result.push(eval_poly_at(poly, new arithmetic(i)).val);
        }
        return result;
    }
    me.repair = repair;

    function transpose(xs) {
        var ys = [];
        for (var i = 0; i < xs[0].length; i++) {
            var y = [];
            for (var j = 0; j < xs.length; j++) {
                y.push(xs[j][i]);
            }
            ys.push(y);
        }
        return ys;
    }

    // Extends a list of bytearrays
    // eg. extend_chunks([map(ord, 'hello'), map(ord, 'world')], 2)
    // n is the number of redundant error-correction chunks to add
    function extend_chunks(data, n, arithmetic) {
        arithmetic = arithmetic || Galois;
        var o = [];
        for (var i = 0; i < data[0].length; i++) {
            o.push(extend(data.map(function(x) { return x[i]; }), n, arithmetic));
        }
        return transpose(o);
    }
    me.extend_chunks = extend_chunks;

    // Repairs a list of bytearrays. Use null in place of a missing array.
    // Individual arrays can contain some missing or erroneous data.
    function repair_chunks(data, datasize, arithmetic) {
        arithmetic = arithmetic || Galois;
        var first_nonzero = 0;
        while (data[first_nonzero] == null) {
            first_nonzero++;
        }
        var i;
        for (i = 0; i < data.length; i++) {
            if (data[i] == null) {
                data[i] = new Array(data[first_nonzero].length);
            }
        }
        var o = [];
        for (i = 0; i < data[0].length; i++) {
            o.push(repair(data.map(function(x) { return x[i]; }), datasize, arithmetic));
        }
        return transpose(o);
    }
    me.repair_chunks = repair_chunks;

    // Extends either a bytearray or a list of bytearrays or a list of lists...
    // Used in the cubify method to expand a cube in all dimensions
    function deep_extend_chunks(data, n, arithmetic) {
        arithmetic = arithmetic || Galois;
        if (!(data[0] instanceof Array)) {
            return extend(data, n, arithmetic)
        } else {
            var o = [];
            for (var i = 0; i < data[0].length; i++) {
                o.push(deep_extend_chunks(
                            data.map(function(x) { return x[i]; }), n, arithmetic));
            }
            return transpose(o);
        }
    }
    me.deep_extend_chunks = deep_extend_chunks;

    function isObject(o) {
        return typeof o == 'object' || typeof o == 'function';
    }
    if (typeof define == 'function' && typeof define.amd == 'object' && define.amd) {
        define(function() {
            return me;
        });
    } else {
        done = 0
        try {
            if (isObject(module)) { module.exports = me; }
            else (isObject(window) ? window : this).Erasure = me;
        }
        catch(e) {
            (isObject(window) ? window : this).Erasure = me;
        }
    }
}.call(this));
