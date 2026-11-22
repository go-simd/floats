//go:build arm64

package floats

import (
	"math/rand"
	"testing"
)

// TestCMulToKernelRuns calls the NEON kernel itself: within the bound for every number of
// a step, and nothing written past floor(n/4) steps.
func TestCMulToKernelRuns(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, n := range []int{4, 8, 12, 16, 20} {
		a, b := cmulRand(rng, n+2, 3), cmulRand(rng, n+2, 3)
		d := make([]complex128, n+2)
		for i := range d {
			d[i] = complex(mirrorSentinel, 0)
		}
		cmulToNEON(f64ptr(&d[0]), f64ptr(&a[0]), f64ptr(&b[0]), n/4)
		for i := 0; i < n+2; i++ {
			if i < n && !cmulOK(d[i], a[i], b[i]) {
				t.Fatalf("n=%d kernel d[%d] = %v, want %v", n, i, d[i], a[i]*b[i])
			}
			if i >= n && d[i] != complex(mirrorSentinel, 0) {
				t.Fatalf("n=%d: kernel wrote d[%d], past its last full step", n, i)
			}
		}
	}
}
