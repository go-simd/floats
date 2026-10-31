package floats

import "math"

// # Elementwise products, combinations, clamping and the infinity norm
//
// Mul, Axpby3, Clamp and MaxAbs are the vector kernels of splitting solvers
// (ADMM/OSQP-style quadratic programming): the elementwise product with a
// diagonal, the three-term update, the projection onto a box and the ∞-norm of
// a residual. Like the updates of axpy.go they work on the first len(a) (len(v))
// elements, panic if another slice is shorter, and allow dst to be the very same
// slice as an input (partially overlapping slices are not supported).
//
// Bit-identical on every architecture, whatever the CPU features found at run
// time (the payload of a NaN produced by an arithmetic operation aside):
//
//   - Mul is one IEEE multiply per element.
//   - Axpby3 is FUSED: dst[i] = math.FMA(gamma, c[i], math.FMA(beta, b[i], alpha*a[i])).
//     One rounding for the product alpha*a[i], one for each fused step: two
//     FMAs and one multiply per element, VFMADD on amd64 (AVX2+FMA), VFMLA on
//     arm64, the math.FMA loop elsewhere, all the same bits. It differs from the
//     unfused alpha*a[i] + beta*b[i] + gamma*c[i] by a few ULP at most (the fused
//     form rounds less); the tests compare the two with that tolerance.
//   - Clamp and MaxAbs involve no rounding at all and are exact.

// Mul computes dst[i] = a[i]*b[i] for i < len(a). Elements of dst and b past
// len(a) are left untouched. It panics if len(b) < len(a) or len(dst) < len(a).
func Mul(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Mul: len(b) or len(dst) < len(a)")
	}
	mul(dst[:len(a)], a, b[:len(a)])
}

// Axpby3 computes the three-term combination
//
//	dst[i] = alpha*a[i] + beta*b[i] + gamma*c[i]
//
// for i < len(a), fused: dst[i] = math.FMA(gamma, c[i], math.FMA(beta, b[i],
// alpha*a[i])) — see the package notes above. It panics if len(b), len(c) or
// len(dst) is less than len(a).
func Axpby3(dst []float64, alpha float64, a []float64, beta float64, b []float64, gamma float64, c []float64) {
	if len(b) < len(a) || len(c) < len(a) || len(dst) < len(a) {
		panic("floats: Axpby3: len(b), len(c) or len(dst) < len(a)")
	}
	n := len(a)
	axpby3(dst[:n], alpha, a, beta, b[:n], gamma, c[:n])
}

// Clamp projects v onto the box [lo, hi] elementwise, min(max(v[i], lo[i]), hi[i]),
// for i < len(v):
//
//	r := v[i]; if lo[i] > r { r = lo[i] }; if hi[i] < r { r = hi[i] }; dst[i] = r
//
// Exact semantics, identical on every architecture:
//
//   - a NaN v[i] propagates unchanged (same bits, whatever its sign);
//   - a NaN bound is ignored, as if that side were infinite;
//   - lo[i] > hi[i] gives hi[i] (the upper bound wins);
//   - signed zeros are compared as equal, so Clamp(-0, +0, hi) keeps -0 and
//     Clamp(+0, -0, hi) keeps +0: the result is v[i] unless a strict bound
//     replaces it.
//
// It panics if len(lo), len(hi) or len(dst) is less than len(v). For one-sided
// bounds pass ±Inf.
func Clamp(dst, v, lo, hi []float64) {
	if len(lo) < len(v) || len(hi) < len(v) || len(dst) < len(v) {
		panic("floats: Clamp: len(lo), len(hi) or len(dst) < len(v)")
	}
	n := len(v)
	clamp(dst[:n], v, lo[:n], hi[:n])
}

// MaxAbs returns the infinity norm max |v[i]|, and 0 for an empty slice. If any
// element is NaN (of either sign) it returns NaN (math.NaN()); ±Inf elements
// give +Inf.
func MaxAbs(v []float64) float64 {
	r := maxAbs(v)
	if r != r {
		return math.NaN()
	}
	return r
}

// ---- portable reference kernels ----
//
// The oracle the tests compare the SIMD kernels against, and the kernel on
// every architecture without one. The callers have already sliced every slice
// to the common length.

func mulRef(dst, a, b []float64) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v * b[i]
	}
}

func axpby3Ref(dst []float64, alpha float64, a []float64, beta float64, b []float64, gamma float64, c []float64) {
	b, c, dst = b[:len(a)], c[:len(a)], dst[:len(a)]
	for i, v := range a {
		t := alpha * v // rounded on its own: not part of a fused add
		t = math.FMA(beta, b[i], t)
		dst[i] = math.FMA(gamma, c[i], t)
	}
}

func clampRef(dst, v, lo, hi []float64) {
	lo, hi, dst = lo[:len(v)], hi[:len(v)], dst[:len(v)]
	for i, x := range v {
		if l := lo[i]; l > x {
			x = l
		}
		if h := hi[i]; h < x {
			x = h
		}
		dst[i] = x
	}
}

const absMask = 1<<63 - 1

// maxAbsRef is the integer-order maximum of the sign-cleared bit patterns:
// for non-negative floats the order of the bits is the order of the values, +Inf
// is 0x7ff0…, and every NaN is above it, so a NaN anywhere makes the maximum a
// NaN. Four independent lanes keep the compare chains short.
func maxAbsRef(v []float64) float64 {
	var m0, m1, m2, m3 uint64
	i := 0
	for ; i+4 <= len(v); i += 4 {
		m0 = max(m0, math.Float64bits(v[i])&absMask)
		m1 = max(m1, math.Float64bits(v[i+1])&absMask)
		m2 = max(m2, math.Float64bits(v[i+2])&absMask)
		m3 = max(m3, math.Float64bits(v[i+3])&absMask)
	}
	for ; i < len(v); i++ {
		m0 = max(m0, math.Float64bits(v[i])&absMask)
	}
	return math.Float64frombits(max(max(m0, m1), max(m2, m3)))
}
