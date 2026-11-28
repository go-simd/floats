package floats

import (
	"math/rand"
	"testing"
)

// BenchmarkAllPass3Split32 times one slot of 34 rows (the PS hybrid band count of HE-AAC v2)
// through the float32 kernel, its portable reference and the float64 AllPass3 it is meant to
// replace; the plan-tick form is the one a decoder calls per slot.
func BenchmarkAllPass3Split32(b *testing.B) {
	const n = 34
	f := newAP32Fix(rand.New(rand.NewSource(1)), n, 0)
	b.Run("kernel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AllPass3Split32(f.out, f.in, &f.t, &f.old, &f.cur)
		}
	})
	b.Run("portable", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			allPass3Split32Ref(n, f.out, f.in, &f.t, &f.old, &f.cur)
		}
	})
	m := newAP32Mem(rand.New(rand.NewSource(2)), n, n, 1)
	p, err := m.plan()
	if err != nil {
		b.Fatal(err)
	}
	b.Run("plan-tick", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.RunTick(i)
		}
	})
	g := newAPFix(rand.New(rand.NewSource(1)), n, 0, 0)
	b.Run("float64-AllPass3", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AllPass3(g.out, g.in, &g.t, &g.old, &g.cur)
		}
	})
}
