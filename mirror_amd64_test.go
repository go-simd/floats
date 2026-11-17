//go:build amd64

package floats

import "testing"

// TestMirrorCombinePortable runs the oracle with the kernels off: the Go loops alone.
func TestMirrorCombinePortable(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	checkMirror(t, mirrorLens())
}

// TestMirrorCombineKernelRuns fails on a CPU that has AVX2+FMA if the kernel is not
// the one taking the leading values of k (a dispatch that silently fell back).
func TestMirrorCombineKernelRuns(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA")
	}
	n := 9
	z, w1, w2 := make([]float64, 2*n), make([]float64, 2*n), make([]float64, 2*n)
	for i := range z {
		z[i], w1[i], w2[i] = float64(i)+0.3, 0.7, 0.2
	}
	d1, d2 := make([]float64, 2*n), make([]float64, 2*n)
	mirrorCombineAVX2(d1, z, w1, w2)
	mirrorTail(d2, z, w1, w2, n, 1)
	for k := 1; k < 1+2*((n-1)/2); k++ {
		if d1[2*k] == 0 && d1[2*k+1] == 0 {
			t.Fatalf("kernel left k=%d unwritten", k)
		}
	}
}
