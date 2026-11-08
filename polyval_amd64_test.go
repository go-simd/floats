//go:build amd64

package floats

import (
	"math"
	"testing"
)

// TestPolyvalAMD64Paths drives Polyval and Chebval down both amd64 kernels, AVX2
// and SSE2 (by toggling hasAVX2FMA, as the other dispatch tests do), and compares
// each with the portable loop bit for bit.
func TestPolyvalAMD64Paths(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	t.Logf("AVX2+FMA available: %v (the avx2 lane runs the SSE2 kernel when false)", saved)
	paths := []struct {
		name string
		on   bool
	}{{"sse2", false}, {"avx2", saved}}
	for _, p := range paths {
		hasAVX2FMA = p.on
		for k, cs := range pvCases() {
			n := len(cs.x)
			got, want := make([]float64, n), make([]float64, n)
			pvRun(cs.op, got, cs.x, cs.c)
			pvRef(cs.op, want, cs.x, cs.c)
			for j := range got {
				if !sameResult(got[j], want[j]) {
					t.Fatalf("%s case %d op %d nc %d n %d: [%d] %016x vs %016x", p.name, k, cs.op, len(cs.c), n, j,
						math.Float64bits(got[j]), math.Float64bits(want[j]))
				}
			}
		}
	}
}
