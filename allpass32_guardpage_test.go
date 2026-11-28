//go:build unix

package floats

import (
	"math/rand"
	"testing"
)

// TestAllPass3Split32GuardPages runs the kernel with every one of its 27 operands ending
// exactly at a page boundary followed by an inaccessible page (and, separately, starting
// right after one), for every row count 1..40: a load or store past the last row (the masked
// tail of AVX2 included), or before the first, would fault. The result is checked against
// the bound.
func TestAllPass3Split32GuardPages(t *testing.T) {
	var pages [27][]float32
	for i := range pages {
		pages[i] = guardedPage32(t)
	}
	rng := rand.New(rand.NewSource(9))
	for n := 1; n <= 40; n++ {
		for _, atEnd := range []bool{true, false} {
			f := &ap32Fix{n: n}
			for i, p := range f.ops() {
				if atEnd {
					*p = pages[i][len(pages[i])-n:]
				} else {
					*p = pages[i][:n]
				}
			}
			src := newAP32Fix(rng, n, 1)
			for i, p := range f.ops() {
				copy(*p, *src.ops()[i])
			}
			AllPass3Split32(f.out, f.in, &f.t, &f.old, &f.cur)
			for k := 0; k < n; k++ {
				if _, msg := ap32Bound(f, src, k); msg != "" {
					t.Fatalf("guard n=%d atEnd=%v: %s", n, atEnd, msg)
				}
			}
		}
	}
}
