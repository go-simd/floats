//go:build unix

package floats

import "testing"

// TestMulAddRows32GuardPages runs the kernel with dst, a and b ending exactly at a page
// boundary followed by an inaccessible page (and, separately, starting right after one),
// for every column count 1..80 and several row counts and strides: a block loaded or
// stored past the last element would fault. Values stay small integers, so the
// result is exact and must equal the plain loop.
func TestMulAddRows32GuardPages(t *testing.T) {
	pd, pa, pb := guardedPage32(t), guardedPage32(t), guardedPage32(t)
	for n := 1; n <= 80; n++ {
		for _, rows := range []int{1, 2, 5} {
			for _, ld := range []int{n, n + 3, 0} {
				na, nb := (rows-1)*ld+n, (rows-1)*(n+1)+n
				for _, atEnd := range []bool{true, false} {
					cut := func(p []float32, m int) []float32 {
						if atEnd {
							return p[len(p)-m:]
						}
						return p[:m]
					}
					d, a, b := cut(pd, n), cut(pa, na), cut(pb, nb)
					for i := range d {
						d[i] = float32(i) - 3
					}
					for i := range a {
						a[i] = float32(i%7) + 1
					}
					for i := range b {
						b[i] = float32(i%5) - 2
					}
					want := append([]float32(nil), d...)
					for k := 0; k < rows; k++ {
						for j := range want {
							want[j] += a[k*ld+j] * b[k*(n+1)+j]
						}
					}
					MulAddRows32(d, a, ld, b, n+1, rows)
					for j := range d {
						if d[j] != want[j] {
							t.Fatalf("guard n=%d rows=%d ld=%d atEnd=%v: element %d: %v, want %v", n, rows, ld, atEnd, j, d[j], want[j])
						}
					}
				}
			}
		}
	}
}
