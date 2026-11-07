//go:build linux || darwin

package floats

import "testing"

// TestExp32NEONStaysInside fences src and dst of the NEON exp kernel
// (fences32 needs mmap/mprotect, hence the build tag of fence_test.go).
func TestExp32NEONStaysInside(t *testing.T) {
	for n := 8; n <= 64; n += 8 {
		for _, s := range fences32(t, n) {
			for _, d := range fences32(t, n) {
				exp32CRNEON(&d[0], &s[0], n, expCRKN)
			}
		}
	}
}

// TestLog32NEONStaysInside fences src and dst of the NEON log kernel.
func TestLog32NEONStaysInside(t *testing.T) {
	for n := 8; n <= 64; n += 8 {
		for _, s := range fences32(t, n) {
			for _, d := range fences32(t, n) {
				log32CRNEON(&d[0], &s[0], n, logCRKN)
			}
		}
	}
}
