//go:build unix

package floats

import "testing"

// TestMulAddRowsGuardPages runs the kernel with dst, a and b ending exactly at a page
// boundary followed by an inaccessible page (and, separately, starting right after one),
// for every column count 0..40 and several row counts and strides: a block loaded or
// stored past the last element would fault.
func TestMulAddRowsGuardPages(t *testing.T) {
	pd, pa, pb := guardedPage(t), guardedPage(t), guardedPage(t)
	for n := 1; n <= 40; n++ {
		for _, rows := range []int{1, 2, 5} {
			for _, ld := range []int{n, n + 3, 0} {
				na, nb := (rows-1)*ld+n, (rows-1)*(n+1)+n
				for _, atEnd := range []bool{true, false} {
					cut := func(p []float64, m int) []float64 {
						if atEnd {
							return p[len(p)-m:]
						}
						return p[:m]
					}
					d, a, b := cut(pd, n), cut(pa, na), cut(pb, nb)
					for i := range d {
						d[i] = float64(i) - 3
					}
					for i := range a {
						a[i] = float64(i)*0.25 + 1
					}
					for i := range b {
						b[i] = 1 / float64(i+1)
					}
					want := mulRowsWant(d, a, ld, b, n+1, rows)
					MulAddRows(d, a, ld, b, n+1, rows)
					if i := mulRowsDiff(d, want); i >= 0 {
						t.Fatalf("guard n=%d rows=%d ld=%d atEnd=%v: element %d differs", n, rows, ld, atEnd, i)
					}
				}
			}
		}
	}
}
