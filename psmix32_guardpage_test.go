//go:build unix

package floats

import (
	"math/rand"
	"testing"
)

// TestPSMixPlan32GuardPages runs the kernel with every plane ending exactly at a page boundary
// followed by an inaccessible page (and, separately, starting right after one), with exactly the
// maxEnd+7 elements the contract requires: a load or store past that, or before the first
// element, would fault. Run lengths 1..24 alone, and a slot of short runs. The result is
// checked against the bound.
func TestPSMixPlan32GuardPages(t *testing.T) {
	var pages [14][]float32
	for i := range pages {
		pages[i] = guardedPage32(t)
	}
	rng := rand.New(rand.NewSource(10))
	cases := [][]PSMixRun{}
	for n := 1; n <= 24; n++ {
		cases = append(cases, []PSMixRun{{0, n, 1, 0}}, []PSMixRun{{2, 2 + n, 0, 2}})
	}
	var slot []PSMixRun
	for row, b := 0, 0; row < 70; b++ {
		n := 1 + rng.Intn(6)
		slot = append(slot, PSMixRun{row, row + n, b % 4, b % 3})
		row += n
	}
	cases = append(cases, slot)
	for _, runs := range cases {
		for _, atEnd := range []bool{true, false} {
			f := &psmFix{nBands: 4, runs: runs}
			f.finish(rng, 1)
			src := f.clone()
			for i, p := range f.planes() {
				if *p == nil {
					continue
				}
				if atEnd {
					*p = pages[i][len(pages[i])-f.need:]
				} else {
					*p = pages[i][:f.need]
				}
				copy(*p, *src.planes()[i])
			}
			f.call(psmPublic)
			if _, msg := psmCheck(f, src); msg != "" {
				t.Fatalf("guard runs=%v atEnd=%v: %s", runs, atEnd, msg)
			}
		}
	}
}
