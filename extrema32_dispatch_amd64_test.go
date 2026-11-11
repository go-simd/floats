//go:build amd64

package floats

import "testing"

// TestFloat32ExtremaDispatchAmd64 runs the float32 Max/Min suites on both amd64
// paths: the AVX2 kernels (from 32 elements, behind hasAVX2FMA) and the SSE2
// kernels (every length with the gate off, the CPUs without AVX2+FMA, #31).
// Lengths 1..100 cross every block, fold and tail of both; the AVX2 run is skipped
// (CI's amd64 runners all have AVX2+FMA, as the other dispatch tests assume).
func TestFloat32ExtremaDispatchAmd64(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	for _, on := range []bool{false, true} {
		hasAVX2FMA = on
		name := "sse2"
		if on {
			name = "avx2"
		}
		for sub, fn := range map[string]func(*testing.T){
			"NaN":     TestFloat32ExtremaNaNEveryPosition,
			"Zeros":   TestFloat32ExtremaSignedZeros,
			"Random":  TestFloat32ExtremaRandom,
			"Planted": TestFloat32ExtremaPlantedExtremum,
			"Table":   TestFloat32ExtremaNumPyZeroTable,
		} {
			t.Run(name+"/"+sub, fn)
		}
	}
}
