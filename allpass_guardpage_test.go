//go:build unix

package floats

import (
	"math/rand"
	"testing"
	"unsafe"
)

// TestAllPass3GuardPages runs the kernel with every one of its fifteen operands ending
// exactly at a page boundary followed by an inaccessible page (and, separately, starting
// right after one), for every row count 1..40: a load or store past the last row, or
// before the first, would fault.
func TestAllPass3GuardPages(t *testing.T) {
	var pages [15][]float64
	for i := range pages {
		pages[i] = guardedPage(t)
	}
	rng := rand.New(rand.NewSource(9))
	for n := 1; n <= 40; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(i, elem int) unsafe.Pointer {
				p := pages[i]
				if atEnd {
					return unsafe.Pointer(&p[len(p)-elem*n])
				}
				return unsafe.Pointer(&p[0])
			}
			cs := func(i int) []complex128 { return unsafe.Slice((*complex128)(cut(i, 2)), n) }
			fs := func(i int) []float64 { return unsafe.Slice((*float64)(cut(i, 1)), n) }
			f := &apFix{n: n}
			f.out, f.in = cs(0), cs(1)
			f.t = AllPassRows{Phi: cs(2), Q0: cs(3), Q1: cs(4), Q2: cs(5), C0: fs(6), C1: fs(7), C2: fs(8)}
			f.old = [3][]complex128{cs(9), cs(10), cs(11)}
			f.cur = [3][]complex128{cs(12), cs(13), cs(14)}
			src := newAPFix(rng, n, 1, 0)
			copy(f.in, src.in)
			copy(f.t.Phi, src.t.Phi)
			copy(f.t.Q0, src.t.Q0)
			copy(f.t.Q1, src.t.Q1)
			copy(f.t.Q2, src.t.Q2)
			copy(f.t.C0, src.t.C0)
			copy(f.t.C1, src.t.C1)
			copy(f.t.C2, src.t.C2)
			for l := range f.old {
				copy(f.old[l], src.old[l])
			}
			AllPass3(f.out, f.in, &f.t, &f.old, &f.cur)
			allPass3Ref(n, src.out, src.in, &src.t, &src.old, &src.cur)
			if d := f.diff(src); d != "" {
				t.Fatalf("guard n=%d atEnd=%v: %s", n, atEnd, d)
			}
		}
	}
}
