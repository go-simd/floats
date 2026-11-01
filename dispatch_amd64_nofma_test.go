//go:build amd64

package floats

import (
	"math"
	"math/rand"
	"testing"
)

// TestDispatchAMD64NoFMA drives the unfused kernels and the pairwise block down
// both amd64 paths, the AVX2 kernel and the portable fallback, by toggling
// hasAVX2FMA. They must agree bit for bit.
func TestDispatchAMD64NoFMA(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	rng := rand.New(rand.NewSource(96))
	check := func(label string) {
		for _, n := range []int{0, 1, 3, 4, 5, 8, 15, 16, 17, 33, 100, 128, 129, 257, 1000} {
			a, b, c := randF64(n, rng), randF64(n, rng), randF64(n, rng)
			got, want := make([]float64, n), make([]float64, n)

			y := append([]float64(nil), b...)
			w := append([]float64(nil), b...)
			axpyNoFMARef(0.3, a, w)
			AxpyNoFMA(0.3, a, y)
			checkSame(t, label+" axpyNoFMA", y, w, n, 0)

			copy(y, b)
			copy(w, b)
			xpbyNoFMARef(-1.7, a, w)
			XpbyNoFMA(a, -1.7, y)
			checkSame(t, label+" xpbyNoFMA", y, w, n, 0)

			axpby3NoFMARef(want, 0.3, a, -1.7, b, 2.1, c)
			Axpby3NoFMA(got, 0.3, a, -1.7, b, 2.1, c)
			checkSame(t, label+" axpby3NoFMA", got, want, n, 0)

			axpyDiffNoFMARef(want, a, 0.9, b, c)
			AxpyDiffNoFMA(got, a, 0.9, b, c)
			checkSame(t, label+" axpyDiffNoFMA", got, want, n, 0)

			if g, w := SumPairwise(a), pwInit+pairwiseRef(a); math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("%s SumPairwise n=%d: %016x vs %016x", label, n, math.Float64bits(g), math.Float64bits(w))
			}
		}
	}

	hasAVX2FMA = false
	check("portable")
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; kernel branch not exercised on this host")
	}
}
