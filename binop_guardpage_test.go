//go:build unix

package floats

import (
	"math"
	"testing"
)

// TestBinopGuardPages runs Add, Sub, Div, Sqrt and ClampScalar on slices that end
// exactly at a page boundary followed by an inaccessible page (and, separately,
// start right after one), for every length 0..67: a vector load or store past
// either end of any operand would fault.
func TestBinopGuardPages(t *testing.T) {
	pa, pb, pd := guardedPage(t), guardedPage(t), guardedPage(t)
	for n := 0; n <= 67; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float64) []float64 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			a, b, dst := cut(pa), cut(pb), cut(pd)
			for i := 0; i < n; i++ {
				a[i], b[i] = float64(i)+0.5, float64(2*i)-1
			}
			want := make([]float64, n)
			addRef(want, a, b)
			Add(dst, a, b)
			checkSame(t, "Add guard", dst, want, n, 0)
			subRef(want, a, b)
			Sub(dst, a, b)
			checkSame(t, "Sub guard", dst, want, n, 0)
			divRef(want, a, b)
			Div(dst, a, b)
			checkSame(t, "Div guard", dst, want, n, 0)
			for i := range a {
				a[i] = math.Abs(a[i])
			}
			sqrtRef(want, a)
			Sqrt(dst, a)
			checkSame(t, "Sqrt guard", dst, want, n, 0)
			clampScalarRef(want, b, -2, 3)
			ClampScalar(dst, b, -2, 3)
			exactBits(t, "ClampScalar guard", dst, want, n, 0)
		}
	}
}
