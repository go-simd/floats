package floats

import (
	"math/rand"
	"testing"
)

// BenchmarkPSMix32 times one slot of the PS mixing stage of HE-AAC v2 (91 hybrid rows in 34
// bands, runs of 1 to 6 rows, about 3.5 on average) through the kernel behind Run and through
// the plain-Go loop it replaces (the portable reference, the baseline).
func BenchmarkPSMix32(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var runs []PSMixRun
	for row, band := 0, 0; row < 91; band++ {
		n := min(1+rng.Intn(6), 91-row)
		runs = append(runs, PSMixRun{row, row + n, band % 34, band % 3})
		row += n
	}
	f := &psmFix{nBands: 34, runs: runs}
	f.finish(rng, 1)
	b.Run("kernel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			f.call(psmPublic)
		}
	})
	b.Run("plain-go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			f.call(psMix32Ref)
		}
	})
}
