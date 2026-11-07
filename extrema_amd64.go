//go:build amd64

package floats

// SSE2 Max/Min kernels (extrema_amd64_gen.go), the amd64 baseline: no gate.
// MAXPD/MINPD return the second operand when both are zeros, so the sign of a
// zero result is unspecified; it is the one case the scalar loop (builtin
// max/min: +0 beats -0 for Max, -0 beats +0 for Min) resolves, at no cost on any
// other result. A NaN anywhere gives NaN (the kernel's own scan).
//
//go:noescape
func maxSSE2(a *float64, n int) float64

//go:noescape
func minSSE2(a *float64, n int) float64

func maxF64(a []float64) float64 {
	if r := maxSSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return maxRef(a)
}

func minF64(a []float64) float64 {
	if r := minSSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return minRef(a)
}
