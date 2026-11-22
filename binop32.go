package floats

import "math"

// # float32 elementwise Add32, Sub32, Mul32, Div32 and Sqrt32
//
// The single-precision siblings of Add, Sub, Mul, Div and Sqrt (binop.go), wanted
// by go-ndarray, which kept SSE2 copies of them in internal/kernels. Like the
// float64 ones they work on the first len(a) (len(src)) elements, panic if another
// slice is shorter, and allow dst to be the very same slice as an input
// (partially overlapping slices are not supported).
//
// Each element is ONE IEEE-754 binary32 operation (add, subtract, multiply,
// divide, square root), no fused instruction involved, so the result is the exact
// correctly rounded one, bit-identical to the scalar Go loop on every
// architecture and whatever CPU features are found at run time (up to the
// payload of a NaN that an operation itself creates), and NumPy's float32 ufuncs.
// Sqrt32 is float32(math.Sqrt(float64(x))): the square root of a float32 computed
// in binary64 and rounded back is the correctly rounded binary32 root.
//
// amd64: AVX2 (VADDPS ..., the gate of the other kernels) ×4 unrolled, SSE2 (the
// baseline) on a CPU without AVX2+FMA. arm64: NEON (VFADD ... on .S4). Elsewhere
// the portable loops.

// Add32 computes dst[i] = a[i] + b[i] for i < len(a). It panics if len(b) <
// len(a) or len(dst) < len(a).
func Add32(dst, a, b []float32) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Add32: len(b) or len(dst) < len(a)")
	}
	add32(dst[:len(a)], a, b[:len(a)])
}

// Sub32 computes dst[i] = a[i] - b[i] for i < len(a). It panics if len(b) <
// len(a) or len(dst) < len(a).
func Sub32(dst, a, b []float32) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Sub32: len(b) or len(dst) < len(a)")
	}
	sub32(dst[:len(a)], a, b[:len(a)])
}

// Mul32 computes dst[i] = a[i] * b[i] for i < len(a). It panics if len(b) <
// len(a) or len(dst) < len(a).
func Mul32(dst, a, b []float32) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Mul32: len(b) or len(dst) < len(a)")
	}
	mul32(dst[:len(a)], a, b[:len(a)])
}

// Div32 computes dst[i] = a[i] / b[i] for i < len(a). It panics if len(b) <
// len(a) or len(dst) < len(a).
func Div32(dst, a, b []float32) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Div32: len(b) or len(dst) < len(a)")
	}
	div32(dst[:len(a)], a, b[:len(a)])
}

// Sqrt32 computes dst[i] = float32(math.Sqrt(float64(src[i]))) for i < len(src):
// the correctly rounded square root, sqrt(-x) = NaN, sqrt(-0) = -0, sqrt(+Inf) =
// +Inf. It panics if len(dst) < len(src).
func Sqrt32(dst, src []float32) {
	if len(dst) < len(src) {
		panic("floats: Sqrt32: len(dst) < len(src)")
	}
	sqrt32(dst[:len(src)], src)
}

// ---- portable reference kernels (the oracle of the tests, and the kernel on
// every architecture without one). The callers have sliced every slice to the
// common length.

func add32Ref(dst, a, b []float32) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v + b[i]
	}
}

func sub32Ref(dst, a, b []float32) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v - b[i]
	}
}

func mul32Ref(dst, a, b []float32) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v * b[i]
	}
}

func div32Ref(dst, a, b []float32) {
	b, dst = b[:len(a)], dst[:len(a)]
	for i, v := range a {
		dst[i] = v / b[i]
	}
}

func sqrt32Ref(dst, src []float32) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = float32(math.Sqrt(float64(v)))
	}
}
