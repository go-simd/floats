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

func maxF32(a []float32) float32 {
	if r := max32SSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return maxRef32(a)
}

func minF32(a []float32) float32 {
	if r := min32SSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return minRef32(a)
}
