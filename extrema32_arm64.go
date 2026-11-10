//go:build arm64

package floats

// NEON float32 Max/Min kernels (extrema32_arm64_gen.go) from 16 elements up, the
// four-chain loop below that. VFMAX/VFMIN propagate NaN and order -0 below +0, so
// no correction is needed. NEON is the arm64 baseline: no gate.
//
//go:noescape
func max32NEON(a *float32, n int) float32

//go:noescape
func min32NEON(a *float32, n int) float32

func maxF32(a []float32) float32 {
	if len(a) < 16 {
		return maxRef32(a)
	}
	return max32NEON(&a[0], len(a))
}

func minF32(a []float32) float32 {
	if len(a) < 16 {
		return minRef32(a)
	}
	return min32NEON(&a[0], len(a))
}
