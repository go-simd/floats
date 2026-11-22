package floats

import (
	"math"
	"math/rand"
	"testing"
)

// tickMem is a planMem whose link rings hold slots[l] slots, as a decorrelator with
// link delays slots[l]-1 does.
func tickMem(rng *rand.Rand, n, inSlots int, slots [3]int) *planMem {
	m := newPlanMem(rng, n, inSlots, 8)
	for l := range m.w {
		m.w[l] = m.w[l][:n*slots[l]]
	}
	return m
}

func (m *planMem) clone() *planMem {
	c := *m
	c.out, c.in = cloneC(m.out), cloneC(m.in)
	for l := range m.w {
		c.w[l] = cloneC(m.w[l])
	}
	return &c
}

func sameC(a, b []complex128) bool {
	for i := range a {
		if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) ||
			math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
			return false
		}
	}
	return len(a) == len(b)
}

// TestAllPass3PlanRunTickMatchesRun: for every tick of a long run, RunTick is bit-identical
// to Run on the slots the documented formulas give, and leaves every ring identical.
func TestAllPass3PlanRunTickMatchesRun(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	slots := [3][]int{{3, 4, 5}, {2, 2, 2}, {2, 7, 3}}
	for _, n := range []int{1, 2, 3, 23, 34, 50} {
		for _, sl := range slots {
			for _, inSlots := range []int{1, 3, 6} {
				for _, delay := range []int{0, inSlots - 1, inSlots / 2} {
					a := tickMem(rng, n, inSlots, [3]int{sl[0], sl[1], sl[2]})
					b := a.clone()
					pa, err := NewAllPass3TickPlan(&a.t, a.out, a.in, a.inStr, a.w, a.wStr, delay)
					if err != nil {
						t.Fatal(err)
					}
					pb, err := b.plan()
					if err != nil {
						t.Fatal(err)
					}
					for tick := 0; tick < 40; tick++ {
						var old, cur [3]int
						for l := range old {
							old[l] = (tick + 1) % sl[l]
							cur[l] = tick % sl[l]
						}
						pa.RunTick(tick)
						pb.Run((tick+inSlots-delay)%inSlots, old, cur)
						if !sameC(pa.Out(), pb.Out()) {
							t.Fatalf("n=%d slots=%v in=%d delay=%d tick %d: out differs", n, sl, inSlots, delay, tick)
						}
						for l := range a.w {
							if !sameC(a.w[l], b.w[l]) {
								t.Fatalf("n=%d slots=%v tick %d: ring %d differs", n, sl, tick, l)
							}
						}
					}
				}
			}
		}
	}
}

// TestAllPass3PlanRunTickAnyTick: no tick, however large or negative, panics, reads or
// writes outside the proved memory (the slack element past each slice stays intact, and
// the slots written are inside the rings).
func TestAllPass3PlanRunTickAnyTick(t *testing.T) {
	rng := rand.New(rand.NewSource(78))
	const n = 23
	m := tickMem(rng, n, 4, [3]int{3, 5, 8})
	guard := complex(12345.5, -678.25)
	fence := func(s []complex128) *complex128 { p := &s[:len(s)+1][len(s)]; *p = guard; return p }
	var fences []*complex128
	fences = append(fences, fence(m.out), fence(m.in))
	for l := range m.w {
		fences = append(fences, fence(m.w[l]))
	}
	p, err := NewAllPass3TickPlan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, 3)
	if err != nil {
		t.Fatal(err)
	}
	ticks := []int{0, 1, -1, -2, 7, math.MaxInt, math.MinInt, math.MaxInt - 1, math.MinInt + 1,
		math.MaxInt32, math.MinInt32, 1 << 40, -(1 << 40)}
	for i := 0; i < 200; i++ {
		ticks = append(ticks, int(rng.Uint64()))
	}
	for _, tk := range ticks {
		p.RunTick(tk)
		for i, f := range fences {
			if *f != guard {
				t.Fatalf("tick %d: memory past slice %d written", tk, i)
			}
		}
	}
}

// TestAllPass3PlanRunTickNegativeReduces: a negative tick reduces as uint(t) does.
func TestAllPass3PlanRunTickNegativeReduces(t *testing.T) {
	rng := rand.New(rand.NewSource(79))
	a := tickMem(rng, 7, 3, [3]int{2, 3, 4})
	b := a.clone()
	pa, _ := NewAllPass3TickPlan(&a.t, a.out, a.in, a.inStr, a.w, a.wStr, 1)
	pb, _ := b.plan()
	sl := [3]uint{2, 3, 4}
	u := uint(math.MaxUint) - 4 // uint(-5)
	var old, cur [3]int
	for l := range old {
		cur[l] = int(u % sl[l])
		old[l] = (cur[l] + 1) % int(sl[l])
	}
	pa.RunTick(-5)
	pb.Run(int((u%3+3-1)%3), old, cur)
	if !sameC(pa.Out(), pb.Out()) {
		t.Fatal("RunTick(-5) is not Run on the slots of uint(-5)")
	}
}

// TestAllPass3TickPlanRefusals: the input delay is refused outside [0, inSlots), and
// accepted at both ends; the refusals of NewAllPass3Plan stand.
func TestAllPass3TickPlanRefusals(t *testing.T) {
	rng := rand.New(rand.NewSource(80))
	m := tickMem(rng, 5, 4, [3]int{2, 3, 4})
	for _, c := range []struct {
		delay int
		ok    bool
	}{{-1, false}, {0, true}, {3, true}, {4, false}, {1 << 30, false}} {
		p, err := NewAllPass3TickPlan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, c.delay)
		if (err == nil) != c.ok || (p == nil) == c.ok {
			t.Errorf("delay %d: err=%v plan=%v, ok want %v", c.delay, err, p != nil, c.ok)
		}
	}
	m.w[1] = m.w[1][:5] // under two slots of 5 rows
	if _, err := NewAllPass3TickPlan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, 0); err == nil {
		t.Error("a link ring of under two slots was accepted")
	}
}

// TestAllPass3PlanRunTickDoesNotAllocate: a tick call allocates nothing.
func TestAllPass3PlanRunTickDoesNotAllocate(t *testing.T) {
	rng := rand.New(rand.NewSource(81))
	m := tickMem(rng, 34, 3, [3]int{3, 4, 5})
	p, err := NewAllPass3TickPlan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, 2)
	if err != nil {
		t.Fatal(err)
	}
	tick := 0
	if a := testing.AllocsPerRun(50, func() { p.RunTick(tick); tick++ }); a != 0 {
		t.Errorf("RunTick allocates %v times", a)
	}
}

func BenchmarkAllPass3PlanTick(b *testing.B) {
	const n = 34
	rng := rand.New(rand.NewSource(1))
	m := tickMem(rng, n, 3, [3]int{3, 4, 5})
	p, err := NewAllPass3TickPlan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, 2)
	if err != nil {
		b.Fatal(err)
	}
	b.Run("RunTick", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.RunTick(i)
		}
	})
	b.Run("Run", func(b *testing.B) {
		sl := [3]int{3, 4, 5}
		for i := 0; i < b.N; i++ {
			var old, cur [3]int
			for l := range old {
				old[l] = (i + 1) % sl[l]
				cur[l] = i % sl[l]
			}
			p.Run((i+3-2)%3, old, cur)
		}
	})
}
