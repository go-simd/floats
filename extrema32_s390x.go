//go:build s390x

package floats

import "golang.org/x/sys/cpu"

// VXE float32 Max/Min kernels (extrema32_s390x_gen.go: VFMAXSB/VFMINSB in Java
// mode, NaN-propagating with -0 below +0) from 16 elements up on a z14 or later
// (vector-enhancements facility 1), the four-chain loop otherwise. hasVXE is a
// var so a test can drive the fallback.
//
//go:noescape
func max32VXE(a *float32, n int) float32

//go:noescape
func min32VXE(a *float32, n int) float32

var hasVXE = cpu.S390X.HasVXE

func maxF32(a []float32) float32 {
	if !hasVXE || len(a) < 16 {
		return maxRef32(a)
	}
	return max32VXE(&a[0], len(a))
}

func minF32(a []float32) float32 {
	if !hasVXE || len(a) < 16 {
		return minRef32(a)
	}
	return min32VXE(&a[0], len(a))
}
