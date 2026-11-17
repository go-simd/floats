//go:build amd64

package floats

import (
	"math"
	"testing"
)

// TestCMulZipRevPortable runs the oracle with the kernel off: the Go loop alone.
func TestCMulZipRevPortable(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	checkZipRev(t, zipRevLens())
}

// TestCMulZipRevKernelRuns compares the kernel's four-number steps with the portable
// loop, which fails if the dispatch fell back or the kernel is wrong.
func TestCMulZipRevKernelRuns(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA")
	}
	m := 12
	x, y, w := make([]float64, m), make([]float64, m), make([]float64, 2*m)
	for i := range x {
		x[i], y[i] = float64(i)+0.3, float64(3*i)-1.7
	}
	for i := range w {
		w[i] = 0.1*float64(i) - 0.9
	}
	d1, d2 := make([]float64, 2*m), make([]float64, 2*m)
	cmulZipRevAVX2(d1, x, y, w)
	zipRevTail(d2, x, y, w, 0)
	for i := range d1 {
		if math.Abs(d1[i]-d2[i]) > 1e-12*(1+math.Abs(d2[i])) {
			t.Fatalf("kernel d1[%d] = %v, portable loop %v", i, d1[i], d2[i])
		}
	}
}
