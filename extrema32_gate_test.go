//go:build loong64 || ppc64le || riscv64 || s390x

package floats

import "testing"

// TestFloat32ExtremaFallback reruns the suite with the CPU-feature gate of the
// float32 Max/Min kernel off, so the fallback branch is covered (and checked) on a
// CPU that has the kernel.
func TestFloat32ExtremaFallback(t *testing.T) {
	restore := setExtrema32Gate(false)
	defer restore()
	for name, fn := range map[string]func(*testing.T){
		"NaN":    TestFloat32ExtremaNaNEveryPosition,
		"Zeros":  TestFloat32ExtremaSignedZeros,
		"Random": TestFloat32ExtremaRandom,
	} {
		t.Run(name, fn)
	}
}
