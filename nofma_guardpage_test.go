//go:build unix

package floats

import (
	"math"
	"testing"
)

// TestNoFMAGuardPages runs the unfused kernels and SumPairwise on slices that end
// exactly at a page boundary followed by an inaccessible page (and, separately,
// start right after one), for every length 0..300: a vector load or store past
// either end of any operand, in the unrolled loop, the single-vector loop or the
// tail, would fault.
func TestNoFMAGuardPages(t *testing.T) {
	pa, pb, pc, pd := guardedPage(t), guardedPage(t), guardedPage(t), guardedPage(t)
	for n := 0; n <= 300; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float64) []float64 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			a, b, c, dst := cut(pa), cut(pb), cut(pc), cut(pd)
			for i := 0; i < n; i++ {
				a[i], b[i], c[i] = float64(i)+0.5, float64(2*i)-1, 1/float64(i+1)
			}
			want := make([]float64, n)

			axpyDiffNoFMARef(want, a, 0.25, b, c)
			AxpyDiffNoFMA(dst, a, 0.25, b, c)
			checkSame(t, "AxpyDiffNoFMA guard", dst, want, n, 0)

			axpby3NoFMARef(want, 0.25, a, -2, b, 1.5, c)
			Axpby3NoFMA(dst, 0.25, a, -2, b, 1.5, c)
			checkSame(t, "Axpby3NoFMA guard", dst, want, n, 0)

			copy(dst, b)
			copy(want, b)
			axpyNoFMARef(0.25, a, want)
			AxpyNoFMA(0.25, a, dst)
			checkSame(t, "AxpyNoFMA guard", dst, want, n, 0)

			copy(dst, b)
			copy(want, b)
			xpbyNoFMARef(0.25, a, want)
			XpbyNoFMA(a, 0.25, dst)
			checkSame(t, "XpbyNoFMA guard", dst, want, n, 0)

			if g, w := SumPairwise(a), pwInit+pairwiseRef(a); math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("SumPairwise guard n=%d atEnd=%v: %v vs %v", n, atEnd, g, w)
			}
		}
	}
}
