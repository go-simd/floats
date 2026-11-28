package floats

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkAllPass3Split32Rows times the bare kernel (no checks) at row counts with and
// without a masked tail.
func BenchmarkAllPass3Split32Rows(b *testing.B) {
	for _, n := range []int{8, 32, 34, 40, 64, 66} {
		f := newAP32Fix(rand.New(rand.NewSource(1)), n, 0)
		b.Run(fmt.Sprint("n=", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				allPass3Split32(n, f.out, f.in, &f.t, &f.old, &f.cur)
			}
		})
	}
}

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
	for _, stride := range []int{n, 40} {
		m := newAP32Mem(rand.New(rand.NewSource(2)), n, stride, 1)
		m.wStr = [3]int{stride, stride, stride}
		for l := range m.w {
			m.w[l] = Split32{m.w[l].Re[:(3+l)*stride], m.w[l].Im[:(3+l)*stride]}
		}
		p, err := m.plan()
		if err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprint("plan-tick/stride=", stride), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.RunTick(i)
			}
		})
	}
	tm := tickMem(rand.New(rand.NewSource(1)), n, 3, [3]int{3, 4, 5})
	tp, err := NewAllPass3TickPlan(&tm.t, tm.out, tm.in, tm.inStr, tm.w, tm.wStr, 2)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("float64-plan-tick", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tp.RunTick(i)
		}
	})
	g := newAPFix(rand.New(rand.NewSource(1)), n, 0, 0)
	b.Run("float64-AllPass3", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AllPass3(g.out, g.in, &g.t, &g.old, &g.cur)
		}
	})
}
