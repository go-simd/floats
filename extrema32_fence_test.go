//go:build (amd64 || arm64) && (linux || darwin)

package floats

import "testing"

// The float32 Max/Min kernels read through raw pointers: a load one element (or
// one vector) past the slice faults here. Each length from 1 to 100 is placed
// flush against the guard page, once ending at it and once starting after one
// (fence32_test.go), with NaN and zero at the edges as well, so every tail path
// is exercised at the fence.
func TestFloat32ExtremaFenced(t *testing.T) {
	for n := 1; n <= extremaMaxLen; n++ {
		for _, a := range fences32(t, n) {
			for i := range a {
				a[i] = float32(i%11) - 5
			}
			checkExtrema32(t, "fence", a)
			a[0], a[n-1] = nan32, negZero32
			checkExtrema32(t, "fence-edges", a)
			a[0], a[n-1] = negZero32, 0
			checkExtrema32(t, "fence-zeros", a)
			a[0], a[n-1] = 0, nan32
			checkExtrema32(t, "fence-nan-last", a)
		}
	}
}
