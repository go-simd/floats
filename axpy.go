package floats

import "math"

// # Elementwise updates (BLAS level 1)
//
// Axpy, Xpby, AxpyDot and Scale update a float64 slice in place. They are the
// inner loops of iterative solvers (conjugate gradient, GMRES), Householder QR
// and LU elimination.
//
// Bit-identical on every architecture: each updated element is computed with
// exactly ONE rounding, as math.FMA does. Axpy stores math.FMA(alpha, x[i], y[i])
// and Xpby stores math.FMA(beta, y[i], x[i]), whether the SIMD kernel (AVX2 FMA
// on amd64, NEON FMLA on arm64) or the portable math.FMA loop runs, so the
// updated slice never depends on the architecture or on the CPU features found
// at run time. Scale is a plain multiply (one rounding as well). The only
// exception is the payload of a NaN result, which IEEE 754 leaves to the
// hardware; NaN-ness itself is always preserved.
//
// Note that this is NOT the same as the unfused y[i] + alpha*x[i] on amd64,
// where Go does not fuse multiply-adds: a caller replacing such a loop may see
// last-bit differences (the fused form is the more accurate one).
//
// AxpyDot's update of y is bit-identical in the same way. Its returned dot
// product is a reduction, so — like Dot — it is accumulated in several
// independent lanes folded at the end rather than left to right, and agrees
// with a sequential sum to within a few ULP of Σ|y[i]·w[i]|, not bit for bit.
//
// Aliasing: y may be the very same slice as x (or w), which updates it in
// place elementwise. Partially overlapping slices (y = x[1:], say) are not
// supported: the vector kernels load several elements before storing any, so
// the result would differ from a sequential loop.

// Axpy computes y[i] += alpha*x[i] for i < len(x), fused: y[i] =
// math.FMA(alpha, x[i], y[i]). Elements of y past len(x) are left untouched. It
// panics if len(y) < len(x).
func Axpy(alpha float64, x, y []float64) {
	if len(y) < len(x) {
		panic("floats: Axpy: len(y) < len(x)")
	}
	axpy(alpha, x, y[:len(x)])
}

// Xpby computes y[i] = x[i] + beta*y[i] for i < len(x), fused: y[i] =
// math.FMA(beta, y[i], x[i]). It is the search-direction update of conjugate
// gradient (p = r + βp). It panics if len(y) < len(x).
func Xpby(x []float64, beta float64, y []float64) {
	if len(y) < len(x) {
		panic("floats: Xpby: len(y) < len(x)")
	}
	xpby(beta, x, y[:len(x)])
}

// AxpyDot computes y[i] += alpha*x[i] for i < len(x) exactly as Axpy does and
// returns Σ y[i]*w[i] over the updated elements, in one pass over the three
// slices — the residual update and its norm in conjugate gradient
// (r -= αAp; rr = r·r is AxpyDot(-α, Ap, r, r)). It panics if len(y) < len(x)
// or len(w) < len(x).
func AxpyDot(alpha float64, x, y, w []float64) float64 {
	if len(y) < len(x) || len(w) < len(x) {
		panic("floats: AxpyDot: len(y) or len(w) < len(x)")
	}
	return axpyDot(alpha, x, y[:len(x)], w[:len(x)])
}

// Scale computes x[i] *= alpha for every element of x.
func Scale(alpha float64, x []float64) { scale(alpha, x) }

// ---- portable reference kernels ----
//
// These are the oracle the tests compare the SIMD kernels against, and the
// kernel on every architecture without one. math.FMA is a single instruction
// where the hardware has FMA (arm64, ppc64le, s390x, riscv64, loong64, and
// amd64 with FMA3) and a correctly rounded software routine elsewhere, so the
// result is the same everywhere. The callers have already sliced y (and w) to
// len(x); reslicing again here lets the compiler drop the bounds checks.

func axpyRef(alpha float64, x, y []float64) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] = math.FMA(alpha, v, y[i])
	}
}

func xpbyRef(beta float64, x, y []float64) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] = math.FMA(beta, y[i], v)
	}
}

// axpyDotRef updates y like axpyRef and accumulates Σ y[i]*w[i] in four
// independent fused lanes (element i goes to lane i%4), summed (l0+l1)+(l2+l3).
func axpyDotRef(alpha float64, x, y, w []float64) float64 {
	n := len(x)
	y, w = y[:n], w[:n]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= n; i += 4 {
		y0 := math.FMA(alpha, x[i], y[i])
		y1 := math.FMA(alpha, x[i+1], y[i+1])
		y2 := math.FMA(alpha, x[i+2], y[i+2])
		y3 := math.FMA(alpha, x[i+3], y[i+3])
		y[i], y[i+1], y[i+2], y[i+3] = y0, y1, y2, y3
		s0 = math.FMA(y0, w[i], s0)
		s1 = math.FMA(y1, w[i+1], s1)
		s2 = math.FMA(y2, w[i+2], s2)
		s3 = math.FMA(y3, w[i+3], s3)
	}
	for ; i < n; i++ {
		v := math.FMA(alpha, x[i], y[i])
		y[i] = v
		s0 = math.FMA(v, w[i], s0)
	}
	return (s0 + s1) + (s2 + s3)
}

func scaleRef(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}
