//go:build amd64

package floats

import (
	"math"
	"testing"
)

// TestSSE2ForcedPath runs the elementwise test suites with the AVX2+FMA gate off,
// which routes Add, Sub, Mul, Div, Sqrt and ClampScalar to the SSE2 kernels of
// sse2_amd64.s on any amd64 host (CI hosts have AVX2, so the CPU's own fallback
// path is never taken there). The same suites that hold the AVX2 kernels to the
// scalar loop bit for bit hold the SSE2 ones to it: random data with specials
// (NaN, ±Inf, ±0, subnormals), every tail length, exact aliasing of dst with an
// input, and guard pages on both sides of every slice.
func TestSSE2ForcedPath(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false

	for _, tc := range []struct {
		name string
		fn   func(*testing.T)
	}{
		{"Binop", TestBinopBitIdentical},
		{"Sqrt", TestSqrtBitIdentical},
		{"ClampScalar", TestClampScalarBitIdentical},
		{"ClampScalarIsClamp", TestClampScalarIsClamp},
		{"Aliasing", TestBinopAliasing},
		{"Lengths", TestBinopLengths},
		{"Mul", TestMulBitIdentical},
		{"ElemAliasing", TestElemAliasing},
		{"ElemLongerSlices", TestElemLongerSlices},
		{"BinopGuardPages", TestBinopGuardPages},
		{"ElemGuardPages", TestElemGuardPages},
	} {
		t.Run(tc.name, tc.fn)
	}
}

// TestSSE2ClampZerosAndNaN pins the operand order of MAXPD/MINPD: signed zeros
// compare equal (the unbounded side keeps v), a NaN v passes through, a NaN
// bound is ignored, in every lane position and in the scalar tail.
func TestSSE2ClampZerosAndNaN(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false

	nan := func() float64 { return math.NaN() }
	for _, lo := range []float64{negZero, 0, nan(), -1} {
		for _, hi := range []float64{negZero, 0, nan(), 1} {
			for n := 1; n <= 11; n++ {
				for pos := 0; pos < n; pos++ {
					for _, x := range []float64{negZero, 0, nan(), -2, 2, 0.5} {
						v := make([]float64, n)
						v[pos] = x
						got, want := make([]float64, n), make([]float64, n)
						ClampScalar(got, v, lo, hi)
						clampScalarRef(want, v, lo, hi)
						exactBits(t, "ClampScalarSSE2", got, want, n, 0)
					}
				}
			}
		}
	}
}
