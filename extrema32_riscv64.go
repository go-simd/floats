//go:build riscv64

package floats

// RVV float32 Max/Min kernels (extrema32_riscv64_gen.go: vfmax.vv with a vmfne NaN
// count) when the CPU reports V, the four-chain loop otherwise.
//
//go:noescape
func max32RVV(a *float32, n int) float32

//go:noescape
func min32RVV(a *float32, n int) float32

func maxF32(a []float32) float32 {
	if !hasRVV {
		return maxRef32(a)
	}
	return max32RVV(&a[0], len(a))
}

func minF32(a []float32) float32 {
	if !hasRVV {
		return minRef32(a)
	}
	return min32RVV(&a[0], len(a))
}
