package floats

import "math"

// # Weighted max-norm
//
// MaxAbsMul is the weighted max-norm of ODE error control (ODEPACK's vmnorm,
// SciPy's lsoda): max_i |v[i]| * w[i]. It rounds once per product and the
// maximum is order independent, so the SIMD kernel and the portable loop agree
// to the bit, signed zeros and NaNs included.

// MaxAbsMul returns max over i of |v[i]|*w[i], folded with C's fmax from +0:
//
//   - a NaN product (a NaN element, or Inf*0) is ignored, as fmax ignores a NaN
//     argument; the result is NaN never (unlike MaxAbs, which propagates it);
//   - the fold starts at +0 and a product replaces the running maximum when it is
//     greater or equal, so a negative product (w < 0) never wins over the seed
//     and the sign of a zero result is that of the last zero product seen, +0 if
//     there is none: exactly the loop
//     `m = fmax(m, |v[i]|*w[i])` with m = +0;
//   - it returns +0 for an empty slice.
//
// It panics if len(w) < len(v). Only the first len(v) elements of w are read.
func MaxAbsMul(v, w []float64) float64 {
	if len(w) < len(v) {
		panic("floats: MaxAbsMul: len(w) < len(v)")
	}
	w = w[:len(v)]
	if r := maxAbsMul(v, w); r != 0 {
		return r
	}
	// A zero maximum: the sign of that zero depends on the order of the
	// products, which the vector kernel does not keep.
	return maxAbsMulRef(v, w)
}

// maxAbsMulRef is the definition: the sequential fmax fold.
func maxAbsMulRef(v, w []float64) float64 {
	w = w[:len(v)]
	m := 0.0
	for i, x := range v {
		p := math.Abs(x) * w[i]
		switch {
		case p != p:
		case m > p:
		default:
			m = p
		}
	}
	return m
}
