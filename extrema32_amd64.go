//go:build amd64

package floats

// SSE2 float32 Max/Min kernels (extrema32_amd64_gen.go), the amd64 baseline: no
// gate. MAXPS/MINPS return the second operand when both are zeros, so the sign of
// a zero result is unspecified; it is the one case the scalar loop (Go's max/min:
// +0 beats -0 for Max, -0 beats +0 for Min) resolves, at no cost on any other
// result. A NaN anywhere gives NaN (the kernel's own scan).
//
//go:noescape
func max32SSE2(a *float32, n int) float32

//go:noescape
func min32SSE2(a *float32, n int) float32

// The AVX2 kernels (extrema32avx2_amd64_gen.go, floats#46) take 32 floats per
// iteration and run from 32 elements up behind hasAVX2FMA, the gate of the other
// AVX2 kernels; on a CPU without it, and below 32 elements, the SSE2 kernels run.
// Same zero-sign rule: a zero result goes to the scalar loop.
//
//go:noescape
func max32AVX2(a *float32, n int) float32

//go:noescape
func min32AVX2(a *float32, n int) float32

func maxF32(a []float32) float32 {
	var r float32
	if hasAVX2FMA && len(a) >= 32 {
		r = max32AVX2(&a[0], len(a))
	} else {
		r = max32SSE2(&a[0], len(a))
	}
	if r != 0 {
		return r
	}
	return maxRef32(a)
}

func minF32(a []float32) float32 {
	var r float32
	if hasAVX2FMA && len(a) >= 32 {
		r = min32AVX2(&a[0], len(a))
	} else {
		r = min32SSE2(&a[0], len(a))
	}
	if r != 0 {
		return r
	}
	return minRef32(a)
}
