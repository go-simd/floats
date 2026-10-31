//go:build unix

package floats

import (
	"math"
	"testing"
)

// TestElemGuardPages runs Mul, Axpby3, Clamp and MaxAbs on slices that end
// exactly at a page boundary followed by an inaccessible page (and, separately,
// start right after one), for every length 0..67: a vector load or store past
// either end of any operand would fault.
func TestElemGuardPages(t *testing.T) {
	pa, pb, pc, pd := guardedPage(t), guardedPage(t), guardedPage(t), guardedPage(t)
	pl, ph := guardedPage(t), guardedPage(t)
	for n := 0; n <= 67; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float64) []float64 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			a, b, c, dst, lo, hi := cut(pa), cut(pb), cut(pc), cut(pd), cut(pl), cut(ph)
			for i := 0; i < n; i++ {
				a[i], b[i], c[i] = float64(i)+0.5, float64(2*i)-1, 1/float64(i+1)
				lo[i], hi[i] = float64(i)-3, float64(i)+3
			}
			want := make([]float64, n)

			mulRef(want, a, b)
			Mul(dst, a, b)
			checkSame(t, "Mul guard", dst, want, n, 0)

			axpby3Ref(want, 0.25, a, -2, b, 1.5, c)
			Axpby3(dst, 0.25, a, -2, b, 1.5, c)
			checkSame(t, "Axpby3 guard", dst, want, n, 0)

			clampRef(want, b, lo, hi)
			Clamp(dst, b, lo, hi)
			exactBits(t, "Clamp guard", dst, want, n, 0)

			got := MaxAbs(b)
			if w := naiveMaxAbs(b); got != w || math.IsNaN(got) {
				t.Fatalf("MaxAbs guard n=%d: %v want %v", n, got, w)
			}
		}
	}
}
