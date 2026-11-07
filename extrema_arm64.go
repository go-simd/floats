//go:build arm64

package floats

// NEON Max/Min kernels (extrema_arm64_gen.go), the arm64 baseline: no gate.
// FMAX/FMIN propagate NaN and order -0 below +0 exactly as Go's builtin max/min
// (and NumPy), so the result needs no fix-up; fewer than 8 elements use the loop.
//
//go:noescape
func maxNEON(a *float64, n int) float64

//go:noescape
func minNEON(a *float64, n int) float64

func maxF64(a []float64) float64 {
	if len(a) < 8 {
		return maxRef(a)
	}
	return maxNEON(&a[0], len(a))
}

func minF64(a []float64) float64 {
	if len(a) < 8 {
		return minRef(a)
	}
	return minNEON(&a[0], len(a))
}
