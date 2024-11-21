package erasure_code

import "fmt"

func init() {
	galoisInit()
}

// Finite fields
// =============

type ZeroDivisionError struct {
}

func (e *ZeroDivisionError) Error() string {
	return "division by zero"
}

// should panic with ZeroDivisionError when dividing by zero
type Field interface {
	Add(b Field) Field
	Sub(b Field) Field
	Mul(b Field) Field
	Div(b Field) Field
	Value() int
	Factory() FieldFactory
}
type FieldFactory interface {
	Construct(v int) Field
}

// per-byte 2^8 Galois field
// Note that this imposes a hard limit that the number of extended chunks can
// be at most 256 along each dimension
type Galois struct {
	v uint8
}

var gexptable [255]uint8
var glogtable [256]uint8

func galoisTpl(a uint8) uint8 {
	r := a ^ (a << 1) // a * (x+1)
	if (a & (1 << 7)) != 0 {
		// would overflow (have an x^8 term); reduce by the AES polynomial,
		// x^8 + x^4 + x^3 + x + 1
		return r ^ 0x1b
	} else {
		return r
	}
}

func galoisInit() {
	var v uint8 = 1
	for i := uint8(0); i < 255; i++ {
		glogtable[v] = i
		gexptable[i] = v
		v = galoisTpl(v)
	}
}

func (a *Galois) Add(_b Field) Field {
	b := _b.(*Galois)
	return &Galois{a.v ^ b.v}
}
func (a *Galois) Sub(_b Field) Field {
	b := _b.(*Galois)
	return &Galois{a.v ^ b.v}
}
func (a *Galois) Mul(_b Field) Field {
	b := _b.(*Galois)
	if a.v == 0 || b.v == 0 {
		return &Galois{0}
	}
	return &Galois{gexptable[(int(glogtable[a.v])+
		int(glogtable[b.v]))%255]}
}
func (a *Galois) Div(_b Field) Field {
	b := _b.(*Galois)
	if b.v == 0 {
		panic(ZeroDivisionError{})
	}
	if a.v == 0 {
		return &Galois{0}
	}
	return &Galois{gexptable[(int(glogtable[a.v])+255-
		int(glogtable[b.v]))%255]}
}
func (a *Galois) Value() int {
	return int(a.v)
}
func (a *Galois) String() string {
	return fmt.Sprintf("%d",a.v)
}

type galoisFactory struct {
}

func GaloisFactory() FieldFactory {
	return &galoisFactory{}
}
func (self *Galois) Factory() FieldFactory {
	return GaloisFactory()
}
func (self *galoisFactory) Construct(v int) Field {
	return &Galois{uint8(v)}
}

// Modular arithmetic class
type modulo struct {
	v uint
	n uint // the modulus
}

func (a *modulo) Add(_b Field) Field {
	b := _b.(*modulo)
	return &modulo{(a.v + b.v) % a.n, a.n}
}

func (a *modulo) Sub(_b Field) Field {
	b := _b.(*modulo)
	return &modulo{(a.v + a.n - b.v) % a.n, a.n}
}

func (a *modulo) Mul(_b Field) Field {
	b := _b.(*modulo)
	return &modulo{(a.v * b.v) % a.n, a.n}
}

func powmod(b uint, e uint, m uint) uint {
	var r uint = 1
	for e > 0 {
		if (e & 1) == 1 {
			r = (r * b) % m
		}
		b = (b * b) % m
		e >>= 1
	}
	return r
}

func (a *modulo) Div(_b Field) Field {
	b := _b.(*modulo)
	return &modulo{(a.v * powmod(b.v, a.n-2, a.n)) % a.n, a.n}
}

func (self *modulo) Value() int {
	return int(self.v)
}

type moduloFactory struct {
	n uint
}

func (self *modulo) Factory() FieldFactory {
	return &moduloFactory{self.n}
}

func (self *moduloFactory) Construct(v int) Field {
	return &modulo{uint(v), self.n}
}

func MakeModuloFactory(n uint) FieldFactory {
	return &moduloFactory{n}
}

func zero(f FieldFactory) Field {
	return f.Construct(0)
}
func one(f FieldFactory) Field {
	return f.Construct(1)
}

// Helper functions
// ================

// Evaluates a polynomial p in little-endian form (e.g. x^2 + 3x + 2 is
// represented as [2, 3, 1]) at coordinate x,
func EvalPolyAt(poly []Field, x Field) Field {
	arithmetic := x.Factory()
	r, xi := zero(arithmetic), one(arithmetic)
	for _, ci := range poly {
		r = r.Add(xi.Mul(ci))
		xi = xi.Mul(x)
	}
	return r
}

// Given p+1 y values and x values with no errors, recovers the original
// p+1 degree polynomial. For example,
// LagrangeInterp({51.0, 59.0, 66.0}, {1, 3, 4}) = {50.0, 0, 1.0}
// (or it would be, if floats were Fields)
func LagrangeInterp(pieces []Field, xs []Field) []Field {
	arithmetic := pieces[0].Factory()
	zero, one := zero(arithmetic), one(arithmetic)

	// `size` is the number of datapoints; the degree of the result polynomial
	// is then `size-1`
	size := len(pieces)

	root := []Field{one} // initially just the polynomial "1"
	// build up the numerator polynomial, `root`, by taking the product of (x-v)
	// (implemented as convolving repeatedly with [-v, 1])
	for _, v := range xs {
		// iterate backward since new root[i] depends on old root[i-1]
		for i := len(root) - 1; i >= 0; i-- {
			root[i] = root[i].Mul(zero.Sub(v))
			if i > 0 {
				root[i] = root[i].Add(root[i-1])
			}
		}
		// polynomial is always monic so save an extra multiply by doing this
		// after
		root = append(root, one)
	}

	// generate per-value numerator polynomials by dividing the master
	// polynomial back by each x coordinate
	nums := make([][]Field, size)
	for i, v := range xs {
		// divide `root` by (x-v) to get a degree size-1 polynomial
		// (i.e. with `size` coefficients)
		num := make([]Field, size)
		// compute the x^0, x^1, ..., x^(p-2) coefficients by long division
		last := one
		num[len(num)-1] = last // still always a monic polynomial
		for j := size - 2; j >= 0; j-- {
			last = root[j+1].Add(last.Mul(v))
			num[j] = last
		}
		nums[i] = num
	}

	// generate denominators by evaluating numerator polys at their x
	denoms := make([]Field, size)
	for i, x := range xs {
		denoms[i] = EvalPolyAt(nums[i], x)
	}

	// generate output polynomial by taking the sum over i of
	// (nums[i] * pieces[i] / denoms[i])
	sum := make([]Field, size)
	for i := range sum {
		sum[i] = zero
	}
	for i, y := range pieces {
		factor := y.Div(denoms[i])
		// add nums[i] * factor to sum, as a vector
		for j := 0; j < size; j++ {
			sum[j] = sum[j].Add(nums[i][j].Mul(factor))
		}
	}
	return sum
}

// Given two linear equations, eliminates the first variable and returns
// the resulting equation.
//
// An equation of the form a_1 x_1 + ... + a_n x_n + b = 0
// is represented as the array [a_1, ..., a_n, b].
func elim(a []Field, b []Field) []Field {
	result := make([]Field, len(a)-1)
	for i := range result {
		result[i] = a[i+1].Mul(b[0]).Sub(b[i+1].Mul(a[0]))
	}
	return result
}

// Given one homogeneous linear equation and the values of all but the first
// variable, solve for the value of the first variable.
//
// For an equation of the form
//     a_1 x_1 + ... + a_n x_n = 0
// pass two arrays, [a_1, ..., a_n] and [x_2, ..., x_n].
func evaluate(coeffs []Field, vals []Field) Field {
	total := zero(coeffs[0].Factory())
	for i, val := range vals {
		total = total.Sub(coeffs[i+1].Mul(val))
	}
	return total.Div(coeffs[0])
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
func SysSolve(eqs [][]Field) []Field {
	arithmetic := eqs[0][0].Factory()
	backEqs := make([][]Field, 1, len(eqs))
	backEqs[0] = eqs[0]

	for len(eqs) > 1 {
		neweqs := make([][]Field, len(eqs)-1)
		for i := 0; i < len(eqs)-1; i++ {
			neweqs[i] = elim(eqs[i], eqs[i+1])
		}
		eqs = neweqs
		// find a row with a nonzero first entry
		i := 0
		for i+1 < len(eqs) && eqs[i][0].Value() == 0 {
			i++
		}
		backEqs = append(backEqs, eqs[i])
	}

	kvals := make([]Field, len(backEqs)+1)
	kvals[len(backEqs)] = one(arithmetic)
	// back-substitute in reverse order
	// (smallest to largest equation)
	for i := len(backEqs) - 1; i >= 0; i-- {
		kvals[i] = evaluate(backEqs[i], kvals[i+1:])
	}

	return kvals[:len(kvals)-1]
}

// Divide two polynomials with nonzero leading terms.
// T should be a field.
func PolyDiv(Q []Field, E []Field) []Field {
	if len(Q) < len(E) {
		return []Field{}
	}
	div := make([]Field, len(Q)-len(E)+1)
	for i := len(div) - 1; i >= 0; i-- {
		factor := Q[len(Q)-1].Div(E[len(E)-1])
		div[i] = factor
		// subtract factor * E * x^i from Q
		Q = Q[:len(Q)-1] // the highest term should cancel
		for j := 0; j < len(E)-1; j++ {
			Q[i+j] = Q[i+j].Sub(factor.Mul(E[j]))
		}
	}
	return div
}

func trySysSolve(eqs [][]Field) (ret []Field, ok bool) {
	defer func() {
		err := recover()
		if err == nil {
			return
		}
		switch err := err.(type) {
		case ZeroDivisionError:
			ret = nil
			ok = false
		default:
			panic(err)
		}
	}()
	return SysSolve(eqs), true
}

type NotEnoughData struct { }
func (self *NotEnoughData) Error() string {
	return "Not enough data!"
}
type TooManyErrors struct { }
func (self *TooManyErrors) Error() string {
	return "Answer doesn't match (too many errors)!"
}

// Given a set of y coordinates and x coordinates, and the degree of the
// original polynomial, determines the original polynomial even if some of the y
// coordinates are wrong. If m is the minimal number of pieces (ie.  degree +
// 1), t is the total number of pieces provided, then the algo can handle up to
// (t-m)/2 errors.
func BerlekampWelchAttempt(pieces []Field, xs []Field, masterDegree int) ([]Field, error) {
	errorLocatorDegree := (len(pieces) - masterDegree - 1) / 2
	arithmetic := pieces[0].Factory()
	zero, one := zero(arithmetic), one(arithmetic)
	// Set up the equations for y[i]E(x[i]) = Q(x[i])
	// degree(E) = errorLocatorDegree
	// degree(Q) = masterDegree + errorLocatorDegree - 1
	eqs := make([][]Field, 2*errorLocatorDegree+masterDegree+1)
	for i := range eqs {
		eq := []Field{}
		x := xs[i]
		piece := pieces[i]
		neg_x_j := zero.Sub(one)
		for j := 0; j < errorLocatorDegree+masterDegree+1; j++ {
			eq = append(eq, neg_x_j)
			neg_x_j = neg_x_j.Mul(x)
		}
		x_j := one
		for j := 0; j < errorLocatorDegree+1; j++ {
			eq = append(eq, x_j.Mul(piece))
			x_j = x_j.Mul(x)
		}
		eqs[i] = eq
	}
	// Solve the equations
	// Assume the top error polynomial term to be one
	errors := errorLocatorDegree
	ones := 1
	var polys []Field
	for errors >= 0 {
		if p, ok := trySysSolve(eqs); ok {
			for i := 0; i < ones; i++ {
				p = append(p, one)
			}
			polys = p
			break
		}
		// caught ZeroDivisionError
		eqs = eqs[:len(eqs)-1]
		for i, eq := range eqs {
			eq[len(eq)-2] = eq[len(eq)-2].Add(eq[len(eq)-1])
			eqs[i] = eq[:len(eq)-1]
		}
		errors--
		ones++
	}
	if errors < 0 {
		return nil, &NotEnoughData{}
	}
	// divide the polynomials...
	split := errorLocatorDegree + masterDegree + 1
	div := PolyDiv(polys[:split], polys[split:])
	corrects := 0
	for i := 0; i < len(xs); i++ {
		if EvalPolyAt(div, xs[i]).Value() == pieces[i].Value() {
			corrects++
		}
	}
	if corrects < masterDegree+errors {
		return nil, &TooManyErrors{}
	}
	return div, nil
}

// Extends a list of integers in [0 ... 255] (if using Galois arithmetic) by
// adding n redundant error-correction values
func Extend(data []int, n int, arithmetic FieldFactory) ([]int, error) {
	size := len(data)

	dataF := make([]Field, size)
	for i, d := range data {
		dataF[i] = arithmetic.Construct(d)
	}

	xs := make([]Field, size)
	for i := range xs {
		xs[i] = arithmetic.Construct(i)
	}

	poly, err := BerlekampWelchAttempt(dataF, xs, size-1)
	if err != nil {
		return nil, err
	}

	for i := 0; i < n; i++ {
		data = append(data,
		int(EvalPolyAt(poly, arithmetic.Construct(size+i)).Value()))
	}
	return data, nil
}

// Repairs a list of integers in [0 ... 255]. Some integers can be erroneous,
// and you can put -1 in place of an integer if you know that a certain
// value is defective or missing. Uses the Berlekamp-Welch algorithm to
// do error-correction
func Repair(data []int, datasize int, arithmetic FieldFactory) ([]int, error) {
	vs := make([]Field, 0, len(data))
	xs := make([]Field, 0, len(data))
	for i, d := range data {
		if d >= 0 {
			vs = append(vs, arithmetic.Construct(d))
			xs = append(xs, arithmetic.Construct(i))
		}
	}

	poly, err := BerlekampWelchAttempt(vs, xs, datasize-1)
	if err != nil {
		return nil, err
	}

	result := make([]int, len(data))
	for i := range result {
		result[i] = int(EvalPolyAt(poly, arithmetic.Construct(i)).Value())
	}
	return result, nil
}

func transpose(d [][]int) [][]int {
	width := len(d[0])
	result := make([][]int, width)
	for i := range result {
		col := make([]int, len(d))
		for j := range col {
			col[j] = d[j][i]
		}
		result[i] = col
	}
	return result
}

func extractColumn(d [][]int, j int) []int {
	result := make([]int, len(d))
	for i, row := range d {
		result[i] = row[j]
	}
	return result
}

// Extends a list of bytearrays
// eg. ExtendChunks([map(ord, 'hello'), map(ord, 'world')], 2)
// n is the number of redundant error-correction chunks to add
func ExtendChunks(data [][]int, n int, arithmetic FieldFactory) ([][]int, error) {
	width := len(data[0])
	o := make([][]int, width)
	for i := 0; i < width; i++ {
		row, err := Extend(extractColumn(data, i), n, arithmetic)
		if err != nil {
			return nil, err
		}
		o[i] = row
	}
	return transpose(o), nil
}

// Repairs a list of bytearrays. Use an empty array in place of a missing array.
// Individual arrays can contain some missing or erroneous data.
func RepairChunks(data [][]int, datasize int, arithmetic FieldFactory) ([][]int, error) {
	var width int
	for _, row := range data {
		if len(row) > 0 {
			width = len(row)
			break
		}
	}
	filledData := make([][]int, len(data))
	for i, row := range data {
		if len(row) == 0 {
			filledData[i] = make([]int, width)
			for j := range filledData[i] {
				filledData[i][j] = -1
			}
		} else {
			filledData[i] = row
		}
	}
	o := make([][]int, width)
	for i := range o {
		row, err := Repair(extractColumn(data, i), datasize, arithmetic)
		if err != nil {
			return nil, err
		}
		o[i] = row
	}
	return transpose(o), nil
}
