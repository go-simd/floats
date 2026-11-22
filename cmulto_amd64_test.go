//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestCMulToPortable runs the oracle with the kernel off: the Go loop alone.
func TestCMulToPortable(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	checkCMulTo(t)
}

// TestCMulToKernelRuns compares the kernel (steps of four, two and one) with the
// portable loop within the bound for every length, and checks that it writes nothing
// past n.
func TestCMulToKernelRuns(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA")
	}
	rng := rand.New(rand.NewSource(8))
	for n := 1; n <= 21; n++ {
		a, b := cmulRand(rng, n+2, 3), cmulRand(rng, n+2, 3)
		d := make([]complex128, n+2)
		for i := range d {
			d[i] = complex(mirrorSentinel, 0)
		}
		cmulToAVX2(f64ptr(&d[0]), f64ptr(&a[0]), f64ptr(&b[0]), n)
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
