//go:build linux || darwin

package floats

import "testing"

// TestLog32AVX2StaysInside fences src and dst of the AVX2 log kernel
// (fences32 needs mmap/mprotect, hence the build tag of fence_test.go).
func TestLog32AVX2StaysInside(t *testing.T) {
	if !useLog32AVX2 {
		t.Skip("no AVX2 here")
	}
	for n := 8; n <= 64; n += 8 {
		for _, s := range fences32(t, n) {
			for _, d := range fences32(t, n) {
				log32CRAVX2(&d[0], &s[0], n, logCRK)
			}
		}
	}
}
