package floats

import "math"

// # Elementwise Add, Sub, Div, Sqrt and ClampScalar
//
// The float64 elementwise kernels go-ndarray kept privately in internal/kernels
// (K1 of the go-scikit performance campaign: a kernel lives once, in the lowest
// library every user can import). Like Mul they work on the first len(a) (len(src),
// len(v)) elements, panic if another slice is shorter, allow dst to be the very
// same slice as an input (partially overlapping slices are not supported), and
// are bit-identical to the scalar Go loop on every architecture and whatever
// CPU features are found at run time: each element is one IEEE-754 operation
// (add, subtract, divide, square root) or, for ClampScalar, a comparison with no
// rounding at all. No fused instruction is involved, so the results are also
// NumPy's (np.add, np.subtract, np.divide, np.sqrt, np.clip with scalar bounds),
// up to the payload of a NaN that an operation itself creates.
//
// amd64 (AVX2, the gate of the other kernels) uses VADDPD / VSUBPD / VDIVPD /
// VSQRTPD, arm64 NEON VFADD / VFSUB / VFDIV / VFSQRT, both ×4 unrolled; the other
// architectures run the portable loops until a kernel is measured faster on real
// silicon.

// Add computes dst[i] = a[i] + b[i] for i < len(a). It panics if len(b) < len(a)
// or len(dst) < len(a).
func Add(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Add: len(b) or len(dst) < len(a)")
	}
	add(dst[:len(a)], a, b[:len(a)])
}

// Sub computes dst[i] = a[i] - b[i] for i < len(a). It panics if len(b) < len(a)
// or len(dst) < len(a).
func Sub(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Sub: len(b) or len(dst) < len(a)")
	}
	sub(dst[:len(a)], a, b[:len(a)])
}

// Div computes dst[i] = a[i] / b[i] for i < len(a). It panics if len(b) < len(a)
// or len(dst) < len(a).
func Div(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Div: len(b) or len(dst) < len(a)")
	}
	div(dst[:len(a)], a, b[:len(a)])
}

// Sqrt computes dst[i] = math.Sqrt(src[i]) for i < len(src): the correctly rounded
// square root, sqrt(-x) = NaN, sqrt(-0) = -0, sqrt(+Inf) = +Inf. It panics if
// len(dst) < len(src).
func Sqrt(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Sqrt: len(dst) < len(src)")
	}
	sqrt(dst[:len(src)], src)
}

// ClampScalar limits every element of v to [lo, hi], like np.clip with scalar
// bounds and like Clamp with the same bounds at every index:
//
//	r := v[i]; if lo > r { r = lo }; if hi < r { r = hi }; dst[i] = r
//
// A NaN v[i] passes through unchanged (same bits); a NaN bound is ignored (that
// side is unbounded); lo > hi gives hi; signed zeros compare equal, so the result
// is v[i] unless a strict bound replaces it. It panics if len(dst) < len(v).
func ClampScalar(dst, v []float64, lo, hi float64) {
	if len(dst) < len(v) {
		panic("floats: ClampScalar: len(dst) < len(v)")
	}
	clampScalar(dst[:len(v)], v, lo, hi)
}

// ---- portable reference kernels (the oracle of the tests, and the kernel on
// every architecture without one). The callers have sliced every slice to the
// common length.

func addRef(dst, a, b []float64) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v + b[i]
	}
}

func subRef(dst, a, b []float64) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v - b[i]
	}
}

func divRef(dst, a, b []float64) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v / b[i]
	}
}

func sqrtRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Sqrt(v)
	}
}

func clampScalarRef(dst, v []float64, lo, hi float64) {
	dst = dst[:len(v)]
	for i, x := range v {
		if lo > x {
			x = lo
		}
		if hi < x {
			x = hi
		}
		dst[i] = x
	}
}
