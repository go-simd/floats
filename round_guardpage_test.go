//go:build unix

package floats

import (
	"math"
	"testing"
)

// TestRoundGuardPages runs every function of round.go on slices that end exactly
// at a page boundary followed by an inaccessible page (and, separately, start
// right after one), for every length 0..67: a vector load or store past either end
// of src or dst would fault.
func TestRoundGuardPages(t *testing.T) {
	ps, pd := guardedPage(t), guardedPage(t)
	for _, op := range roundOps {
		for n := 0; n <= 67; n++ {
			for _, atEnd := range []bool{true, false} {
				cut := func(p []float64) []float64 {
					if atEnd {
						return p[len(p)-n:]
					}
					return p[:n]
				}
				src, dst := cut(ps), cut(pd)
				for i := range src {
					src[i] = (float64(i) - 31.5) * 0.75
				}
				if n > 0 {
					src[n-1] = math.Copysign(0, -1)
				}
				op.f(dst, src)
				checkRound(t, op, "guard", dst, src, n, 0)
			}
		}
	}
}
