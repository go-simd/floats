package floats

import "testing"

// TestKernelsKeepCallerSlicesOnStack: no kernel keeps a pointer, so the
// assembly stubs are go:noescape and a caller's arrays stay on its stack.
// Without the directive every slice passed to a kernel leaks, and each of
// these arrays costs a heap allocation per call (a solver passing its work
// vectors to Dot or AxpyNoFMA gained allocations by adopting the package).
func TestKernelsKeepCallerSlicesOnStack(t *testing.T) {
	var sink float64
	allocs := testing.AllocsPerRun(20, func() {
		var a, b, c, d [64]float64
		for i := range a {
			a[i], b[i], c[i] = float64(i), float64(64-i), 0.5
		}
		var a32, b32 [64]float32
		sink += Dot(a[:], b[:]) + Sum(a[:]) + Distance(a[:], b[:]) + MaxAbs(a[:]) +
			Max(a[:]) + Min(a[:]) + SumPairwise(a[:]) + AxpyDot(0.5, a[:], b[:], c[:])
		sink += float64(Float32Dot(a32[:], b32[:]) + Float32Sum(a32[:]) + Float32Distance(a32[:], b32[:]))
		Axpy(0.5, a[:], d[:])
		AxpyNoFMA(0.5, a[:], d[:])
		Xpby(a[:], 0.5, d[:])
		XpbyNoFMA(a[:], 0.5, d[:])
		Scale(0.5, d[:])
		Add(d[:], a[:], b[:])
		Sub(d[:], a[:], b[:])
		Mul(d[:], a[:], b[:])
		Div(d[:], a[:], c[:])
		Sqrt(d[:], a[:])
		Clamp(d[:], a[:], b[:], c[:])
		ClampScalar(d[:], a[:], 1, 2)
		Axpby3(d[:], 1, a[:], 2, b[:], 3, c[:])
		Axpby3NoFMA(d[:], 1, a[:], 2, b[:], 3, c[:])
		AxpyDiffNoFMA(d[:], a[:], 0.5, b[:], c[:])
		AxpyRowsNoFMA(c[:4], a[:], 8, d[:8])
		Polyval(d[:], a[:], c[:4])
		Chebval(d[:], c[:], b[:4])
		sink += d[7]
	})
	if allocs != 0 {
		t.Errorf("%v heap allocations per call: a kernel stub lets its slices escape", allocs)
	}
	_ = sink
}
