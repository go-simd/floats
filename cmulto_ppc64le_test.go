//go:build ppc64le

package floats

import (
	"math/rand"
	"testing"
)

// TestCMulToKernelRuns calls the kernel itself: within the bound for every length, and
// nothing written past n.
func TestCMulToKernelRuns(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for n := 1; n <= 21; n++ {
		a, b := cmulRand(rng, n+2, 3), cmulRand(rng, n+2, 3)
		d := make([]complex128, n+2)
		for i := range d {
			d[i] = complex(mirrorSentinel, 0)
		}
		cmulToVSX(f64ptr(&d[0]), f64ptr(&a[0]), f64ptr(&b[0]), n)
		for i := 0; i < n+2; i++ {
			if i < n && !cmulOK(d[i], a[i], b[i]) {
				t.Fatalf("n=%d kernel d[%d] = %v, want %v", n, i, d[i], a[i]*b[i])
			}
			if i >= n && d[i] != complex(mirrorSentinel, 0) {
				t.Fatalf("n=%d: kernel wrote d[%d], past n", n, i)
			}
		}
	}
}

// TestCMulToDispatch runs the public function with the kernel forced on and off, below and
// above the threshold, so both branches are covered on every CPU.
func TestCMulToDispatch(t *testing.T) {
	saved := cmulToVSXOn
	defer func() { cmulToVSXOn = saved }()
	rng := rand.New(rand.NewSource(9))
	for _, on := range []bool{false, true} {
		cmulToVSXOn = on
		for _, n := range []int{1, cmulToVSXMin - 1, cmulToVSXMin, cmulToVSXMin + 5, 100} {
			a, b := cmulRand(rng, n, 3), cmulRand(rng, n, 3)
			d := make([]complex128, n)
			CMulTo(d, a, b)
			for i := range d {
				if !cmulOK(d[i], a[i], b[i]) {
					t.Fatalf("on=%v n=%d d[%d] = %v, want %v", on, n, i, d[i], a[i]*b[i])
				}
			}
		}
	}
}
