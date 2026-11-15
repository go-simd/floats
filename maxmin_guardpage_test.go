//go:build unix

package floats

import (
	"math"
	"testing"
)

// TestPairwiseGuardPages runs Maximum and Minimum on slices that end exactly at a
// page boundary followed by an inaccessible page (and, separately, start right
// after one), for every length 0..67: a vector load or store past either end of a,
// b or dst would fault.
func TestPairwiseGuardPages(t *testing.T) {
	pa, pb, pd := guardedPage(t), guardedPage(t), guardedPage(t)
	for _, op := range maxminOps {
		for n := 0; n <= 67; n++ {
			for _, atEnd := range []bool{true, false} {
				cut := func(p []float64) []float64 {
					if atEnd {
						return p[len(p)-n:]
					}
					return p[:n]
				}
				a, b, dst := cut(pa), cut(pb), cut(pd)
				for i := range a {
					a[i] = (float64(i) - 31.5) * 0.75
					b[i] = (31.5 - float64(i)) * 0.75
				}
				if n > 0 {
					a[n-1], b[n-1] = math.Copysign(0, -1), 0
				}
				op.f(dst, a, b)
				checkMaxMin(t, op, "guard", dst, a, b, n, 0)
			}
		}
	}
}
