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

// TestAxpyRowsNoFMAGuardPages: x and y end exactly at an inaccessible page (and
// start right after one), every width 0..130 and row counts 1..5: a block load or
// store past either end, in any of the width stages, would fault.
func TestAxpyRowsNoFMAGuardPages(t *testing.T) {
	px, py := guardedPage(t), guardedPage(t)
	for n := 0; n <= 130; n++ {
		for r := 1; r <= 5; r++ {
			for _, atEnd := range []bool{true, false} {
				cut := func(p []float64, m int) []float64 {
					if atEnd {
						return p[len(p)-m:]
					}
					return p[:m]
				}
				x, y := cut(px, (r-1)*n+n), cut(py, n)
				for i := range x {
					x[i] = float64(i)*0.37 - 11
				}
				a := []float64{0.25, -2, 1.5, 3, -0.5}[:r]
				y0 := make([]float64, n)
				for i := range y0 {
					y0[i] = 1 / float64(i+1)
				}
				want := rowsWant(a, x, n, y0)
				copy(y, y0)
				AxpyRowsNoFMA(a, x, n, y)
				checkSame(t, "AxpyRowsNoFMA guard", y, want, n, 0)
			}
		}
	}
}
