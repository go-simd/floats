//go:build linux || darwin

package floats

import "testing"

// TestLogAVX2StaysInside fences src and dst of the AVX2 float64 log kernel
// (fences needs mmap/mprotect, hence the build tag of fence_test.go).
func TestLogAVX2StaysInside(t *testing.T) {
	if !useLogAVX2 {
		t.Skip("no AVX2+FMA here")
	}
	for n := 4; n <= 64; n += 4 {
		for _, s := range fences(t, n) {
			for _, d := range fences(t, n) {
				logAVX2(&d[0], &s[0], n, logK, &logTab[0])
			}
		}
	}
}
