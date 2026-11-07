//go:build linux || darwin

package floats

import "testing"

// TestExp32AVX2StaysInside fences src and dst of the AVX2 exp kernel
// (fences32 needs mmap/mprotect, hence the build tag of fence_test.go).
func TestExp32AVX2StaysInside(t *testing.T) {
	if !useExp32AVX2 {
		t.Skip("no AVX2+FMA here")
	}
	for n := 8; n <= 64; n += 8 {
		for _, s := range fences32(t, n) {
			for _, d := range fences32(t, n) {
				exp32CRAVX2(&d[0], &s[0], n, expCRK)
			}
		}
	}
}
