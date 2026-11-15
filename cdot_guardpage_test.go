//go:build unix

package floats

import "testing"

// TestCDotCGuardPages runs CDotC with a and b ending exactly at a page boundary followed
// by an inaccessible page (and starting right after one), for every length 0..40 complex
// numbers: a block loaded past the last element would fault.
func TestCDotCGuardPages(t *testing.T) {
	pa, pb := guardedPage(t), guardedPage(t)
	for n := 0; n <= 40; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float64) []float64 {
				if atEnd {
					return p[len(p)-2*n:]
				}
				return p[:2*n]
			}
			a, b := cut(pa), cut(pb)
			for i := range a {
				a[i], b[i] = float64(i)*0.25+1, 1/float64(i+1)
			}
			gr, gi := CDotC(a, b)
			cdotCheck(t, "guard", n, gr, gi, a, b)
		}
	}
}
