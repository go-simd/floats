package floats

import (
	"math/rand"
	"strings"
	"testing"
)

// ap32Mem is the memory of a plan: n rows, rings of slots of stride elements.
type ap32Mem struct {
	n, inStr, inDelay int
	t                 AllPassRows32
	out, in           Split32
	w                 [3]Split32
	wStr              [3]int
}

func newAP32Mem(rng *rand.Rand, n, stride, inDelay int) *ap32Mem {
	f := newAP32Fix(rng, n, 0)
	m := &ap32Mem{n: n, inStr: stride, inDelay: inDelay, t: f.t, out: Split32{make([]float32, n), make([]float32, n)}}
	ring := func(slots, str int) Split32 {
		s := Split32{make([]float32, slots*str), make([]float32, slots*str)}
		for i := range s.Re {
			s.Re[i], s.Im[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
		}
		return s
	}
	m.in = ring(4, stride)
	for l := range m.w {
		m.wStr[l] = stride + l
		m.w[l] = ring(3+l, m.wStr[l])
	}
	return m
}

func (m *ap32Mem) plan() (*AllPass3Plan32, error) {
	return NewAllPass3TickPlan32(&m.t, m.out, m.in, m.inStr, m.w, m.wStr, m.inDelay)
}

// TestAllPass3Plan32RunTickMatches: over many ticks, RunTick computes what AllPass3Split32
// computes on the slots the documentation names, and leaves the rest of the rings alone, for
// strides equal to and wider than n.
func TestAllPass3Plan32RunTickMatches(t *testing.T) {
	rng := rand.New(rand.NewSource(32))
	for _, c := range []struct{ n, stride, delay int }{{1, 1, 0}, {7, 7, 3}, {34, 40, 2}, {50, 50, 1}} {
		a := newAP32Mem(rng, c.n, c.stride, c.delay)
		b := newAP32Mem(rng, c.n, c.stride, c.delay)
		b.t = a.t
		copy(b.in.Re, a.in.Re)
		copy(b.in.Im, a.in.Im)
		for l := range a.w {
			copy(b.w[l].Re, a.w[l].Re)
			copy(b.w[l].Im, a.w[l].Im)
		}
		p, err := a.plan()
		if err != nil {
			t.Fatal(err)
		}
		if p.Rows() != c.n {
			t.Fatalf("Rows = %d, want %d", p.Rows(), c.n)
		}
		for tick := -2; tick < 40; tick++ {
			p.RunTick(tick)
			u := uint(tick)
			var old, cur [3]Split32
			for l := range old {
				m := uint(len(b.w[l].Re) / b.wStr[l])
				cs := u % m
				os := (cs + 1) % m
				s := uint(b.wStr[l])
				old[l] = Split32{b.w[l].Re[os*s : os*s+uint(c.n)], b.w[l].Im[os*s : os*s+uint(c.n)]}
				cur[l] = Split32{b.w[l].Re[cs*s : cs*s+uint(c.n)], b.w[l].Im[cs*s : cs*s+uint(c.n)]}
			}
			k := uint(len(b.in.Re) / b.inStr)
			i := (u%k + k - uint(c.delay)) % k
			s := uint(b.inStr)
			AllPass3Split32(b.out, Split32{b.in.Re[i*s : i*s+uint(c.n)], b.in.Im[i*s : i*s+uint(c.n)]}, &b.t, &old, &cur)
			got, want := p.Out(), b.out
			for r := 0; r < c.n; r++ {
				if got.Re[r] != want.Re[r] || got.Im[r] != want.Im[r] {
					t.Fatalf("n=%d tick %d row %d: plan (%v, %v), call (%v, %v)", c.n, tick, r, got.Re[r], got.Im[r], want.Re[r], want.Im[r])
				}
			}
			for l := range a.w {
				for i := range a.w[l].Re {
					if a.w[l].Re[i] != b.w[l].Re[i] || a.w[l].Im[i] != b.w[l].Im[i] {
						t.Fatalf("n=%d tick %d: link ring %d differs at %d", c.n, tick, l, i)
					}
				}
			}
		}
	}
}

// TestAllPass3Plan32Refusals: every refusal of the constructor, one element on each side
// of the bound where there is one.
func TestAllPass3Plan32Refusals(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	const n = 6
	refused := func(m *ap32Mem, want string) {
		t.Helper()
		p, err := m.plan()
		if err == nil || p != nil || !strings.Contains(err.Error(), want) {
			t.Errorf("want an error containing %q, got plan=%v err=%v", want, p != nil, err)
		}
	}
	if _, err := newAP32Mem(rng, n, n, 0).plan(); err != nil {
		t.Fatalf("a good memory was refused: %v", err)
	}
	m := newAP32Mem(rng, n, n, 0)
	m.t = AllPassRows32{}
	refused(m, "no rows")
	for i := 0; i < 11; i++ {
		m = newAP32Mem(rng, n, n, 0)
		tabs := []*[]float32{&m.t.PhiRe, &m.t.PhiIm, &m.t.Q0Re, &m.t.Q0Im, &m.t.Q1Re, &m.t.Q1Im, &m.t.Q2Re, &m.t.Q2Im, &m.t.C0, &m.t.C1, &m.t.C2}
		if i == 0 {
			// the first table defines n: shorten another one in its place
			i = 1
		}
		*tabs[i] = (*tabs[i])[:n-1]
		refused(m, "table")
	}
	m = newAP32Mem(rng, n, n, 0)
	m.out.Re = m.out.Re[:n-1]
	refused(m, "out holds")
	m = newAP32Mem(rng, n, n, 0)
	m.out.Im = m.out.Im[:n-1]
	refused(m, "out holds")
	m = newAP32Mem(rng, n, n, 0)
	m.inStr = n - 1
	refused(m, "input ring")
	m = newAP32Mem(rng, n, n, 0)
	m.inStr = len(m.in.Re) + 1
	refused(m, "input ring")
	m = newAP32Mem(rng, n, n, 0)
	m.in.Im = m.in.Im[:len(m.in.Im)-1]
	refused(m, "input ring")
	for l := 0; l < 3; l++ {
		m = newAP32Mem(rng, n, n, 0)
		m.wStr[l] = n - 1
		refused(m, "link ring")
		m = newAP32Mem(rng, n, n, 0)
		m.w[l] = Split32{m.w[l].Re[:m.wStr[l]], m.w[l].Im[:m.wStr[l]]} // one slot
		refused(m, "link ring")
		m = newAP32Mem(rng, n, n, 0)
		m.w[l].Im = m.w[l].Im[:len(m.w[l].Im)-1]
		refused(m, "link ring")
		m = newAP32Mem(rng, n, n, 0)
		m.w[l] = Split32{m.w[l].Re[:2*m.wStr[l]], m.w[l].Im[:2*m.wStr[l]]} // two slots: accepted
		if _, err := m.plan(); err != nil {
			t.Errorf("a link ring of two slots was refused: %v", err)
		}
	}
	for _, d := range []int{-1, 4} {
		m = newAP32Mem(rng, n, n, d)
		refused(m, "input delay")
	}
	m = newAP32Mem(rng, n, n, 3)
	if _, err := m.plan(); err != nil {
		t.Errorf("the largest input delay was refused: %v", err)
	}
}

// TestAllPass3Plan32Overlap: every written memory (out and the link rings, both planes)
// against every other operand, the same memory and a one-element overlap are refused, a
// neighbour that only touches is accepted, and the tables and the input may overlap one another.
func TestAllPass3Plan32Overlap(t *testing.T) {
	rng := rand.New(rand.NewSource(34))
	const n = 6
	slots := func(m *ap32Mem) []*[]float32 {
		s := []*[]float32{&m.out.Re, &m.out.Im}
		for l := range m.w {
			s = append(s, &m.w[l].Re, &m.w[l].Im)
		}
		s = append(s, &m.in.Re, &m.in.Im, &m.t.PhiRe, &m.t.PhiIm, &m.t.Q0Re, &m.t.Q0Im, &m.t.Q1Re, &m.t.Q1Im, &m.t.Q2Re, &m.t.Q2Im, &m.t.C0, &m.t.C1, &m.t.C2)
		return s
	}
	const written = 8
	for w := 0; w < written; w++ {
		for o := 0; o < 21; o++ {
			if o == w {
				continue
			}
			for _, c := range []string{"same", "one element", "touching"} {
				m := newAP32Mem(rng, n, n, 0)
				ss := slots(m)
				base := (*ss[w])[:cap(*ss[w])]
				// the operand under test becomes a window of the written one: the whole of it
				// (so that every length check passes), moved by `at` elements
				ln, at := len(*ss[o]), 0
				switch c {
				case "one element":
					at = len(*ss[w]) - 1
				case "touching":
					at = len(*ss[w])
				}
				if at+ln > cap(*ss[w]) {
					// the operand is longer than the room behind the written one: take a longer backing array
					bigger := make([]float32, at+ln)
					copy(bigger, base)
					*ss[w] = bigger[:len(*ss[w])]
					base = bigger
				}
				*ss[o] = base[at : at+ln]
				_, err := m.plan()
				over := c != "touching"
				if over && (err == nil || !strings.Contains(err.Error(), "overlaps")) {
					t.Errorf("written %d / operand %d (%s): %v", w, o, c, err)
				}
				if !over && err != nil && strings.Contains(err.Error(), "overlaps") {
					t.Errorf("written %d / operand %d (%s): touching refused: %v", w, o, c, err)
				}
			}
		}
	}
	// read-only operands may share memory
	m := newAP32Mem(rng, n, n, 0)
	m.t.Q0Re, m.t.Q0Im, m.t.C0 = m.t.PhiRe, m.t.PhiRe, m.t.PhiRe
	m.in = Split32{append([]float32(nil), m.in.Re...), m.in.Re[:len(m.in.Re):len(m.in.Re)]}
	if _, err := m.plan(); err != nil {
		t.Errorf("read-only operands sharing memory were refused: %v", err)
	}
}

// TestAllPass3Plan32RunTickDoesNotAllocate: a tick call allocates nothing.
func TestAllPass3Plan32RunTickDoesNotAllocate(t *testing.T) {
	m := newAP32Mem(rand.New(rand.NewSource(35)), 34, 40, 2)
	p, err := m.plan()
	if err != nil {
		t.Fatal(err)
	}
	tick := 0
	if a := testing.AllocsPerRun(100, func() { p.RunTick(tick); tick++ }); a != 0 {
		t.Errorf("RunTick allocates %v times per call", a)
	}
}
