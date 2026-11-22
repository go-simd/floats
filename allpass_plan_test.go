package floats

import (
	"math/rand"
	"strings"
	"testing"
	"unsafe"
)

// planMem is the memory of a plan of n rows: the tables, out, an input ring and three
// link rings, every slice allocated with one element of slack so that a test can lengthen
// it by one.
type planMem struct {
	t       AllPassRows
	out, in []complex128
	w       [3][]complex128
	inStr   int
	wStr    [3]int
}

func newPlanMem(rng *rand.Rand, n, inSlots, wSlots int) *planMem {
	c := func(m int) []complex128 {
		s := make([]complex128, m+1)
		for i := range s {
			s[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
		}
		return s[:m]
	}
	f := func(m int) []float64 {
		s := make([]float64, m+1)
		for i := range s {
			s[i] = rng.Float64()*2 - 1
		}
		return s[:m]
	}
	return &planMem{
		t:   AllPassRows{Phi: c(n), Q0: c(n), Q1: c(n), Q2: c(n), C0: f(n), C1: f(n), C2: f(n)},
		out: c(n), in: c(n * inSlots),
		w:     [3][]complex128{c(n * wSlots), c(n * wSlots), c(n * wSlots)},
		inStr: n, wStr: [3]int{n, n, n},
	}
}

func (m *planMem) plan() (*AllPass3Plan, error) {
	return NewAllPass3Plan(&m.t, m.out, m.in, m.inStr, m.w, m.wStr)
}

func cloneC(s []complex128) []complex128 { return append([]complex128(nil), s...) }

// TestAllPass3PlanRunMatchesAllPass3: a plan's Run is bit-identical to AllPass3 on the
// slices the slot numbers name, for every slot choice, and writes only the current slots.
func TestAllPass3PlanRunMatchesAllPass3(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, n := range []int{1, 2, 3, 7, 23, 34, 50} {
		m := newPlanMem(rng, n, 5, 6)
		p, err := m.plan()
		if err != nil {
			t.Fatal(err)
		}
		if p.Rows() != n {
			t.Fatalf("Rows() = %d, want %d", p.Rows(), n)
		}
		for trial := 0; trial < 12; trial++ {
			inSlot := rng.Intn(5)
			var old, cur [3]int
			for l := range old {
				old[l] = rng.Intn(6)
				cur[l] = (old[l] + 1 + rng.Intn(5)) % 6
			}
			want := make([]complex128, n)
			ws := [3][]complex128{cloneC(m.w[0]), cloneC(m.w[1]), cloneC(m.w[2])}
			var ro, rc [3][]complex128
			for l := range ro {
				ro[l] = ws[l][old[l]*n : old[l]*n+n]
				rc[l] = ws[l][cur[l]*n : cur[l]*n+n]
			}
			AllPass3(want, m.in[inSlot*n:inSlot*n+n], &m.t, &ro, &rc)
			p.Run(inSlot, old, cur)
			for k := range want {
				if p.Out()[k] != want[k] {
					t.Fatalf("n=%d trial %d: out[%d] = %v, want %v", n, trial, k, p.Out()[k], want[k])
				}
			}
			for l := range ws {
				for i := range ws[l] {
					if m.w[l][i] != ws[l][i] {
						t.Fatalf("n=%d trial %d: link %d ring[%d] = %v, want %v", n, trial, l, i, m.w[l][i], ws[l][i])
					}
				}
			}
		}
	}
}

// TestAllPass3PlanPadding: a stride wider than n (the rows of a slot are the first n of
// stride elements, as in a decoder's fixed-size slot) leaves the padding alone.
func TestAllPass3PlanPadding(t *testing.T) {
	rng := rand.New(rand.NewSource(22))
	const n, stride = 5, 9
	m := newPlanMem(rng, n, 1, 1)
	m.in = make([]complex128, 3*stride)
	cur := [3]int{3, 0, 1}
	for l := range m.w {
		m.w[l] = make([]complex128, 4*stride)
		for i := range m.w[l] {
			m.w[l][i] = complex(7, 7)
		}
	}
	m.inStr, m.wStr = stride, [3]int{stride, stride, stride}
	p, err := m.plan()
	if err != nil {
		t.Fatal(err)
	}
	p.Run(2, [3]int{0, 1, 2}, cur)
	for l := range m.w {
		for s := 0; s < 4; s++ {
			for k := 0; k < stride; k++ {
				written := m.w[l][s*stride+k] != complex(7, 7)
				if want := s == cur[l] && k < n; written != want {
					t.Fatalf("link %d slot %d element %d: written %v, want %v", l, s, k, written, want)
				}
			}
		}
	}
}

// TestAllPass3PlanLengthBounds: every length is refused one below its bound, accepted at
// it and one above (Phi defines the row count, so it is the exception: shorter just means
// fewer rows, longer is refused because the other tables do not follow).
func TestAllPass3PlanLengthBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	const n = 6
	lengthen := map[string]func(m *planMem, d int){
		"Q0":      func(m *planMem, d int) { m.t.Q0 = m.t.Q0[:n+d] },
		"Q1":      func(m *planMem, d int) { m.t.Q1 = m.t.Q1[:n+d] },
		"Q2":      func(m *planMem, d int) { m.t.Q2 = m.t.Q2[:n+d] },
		"C0":      func(m *planMem, d int) { m.t.C0 = m.t.C0[:n+d] },
		"C1":      func(m *planMem, d int) { m.t.C1 = m.t.C1[:n+d] },
		"C2":      func(m *planMem, d int) { m.t.C2 = m.t.C2[:n+d] },
		"out":     func(m *planMem, d int) { m.out = m.out[:n+d] },
		"in ring": func(m *planMem, d int) { m.in = m.in[:n+d] },
		"link 0":  func(m *planMem, d int) { m.w[0] = m.w[0][:2*n+d] },
		"link 1":  func(m *planMem, d int) { m.w[1] = m.w[1][:2*n+d] },
		"link 2":  func(m *planMem, d int) { m.w[2] = m.w[2][:2*n+d] },
	}
	for name, set := range lengthen {
		for _, d := range []int{-1, 0, 1} {
			m := newPlanMem(rng, n, 1, 2)
			set(m, d)
			_, err := m.plan()
			if (d < 0) != (err != nil) {
				t.Errorf("%s at %+d: err = %v", name, d, err)
			}
		}
	}
	for _, d := range []int{-1, 0, 1} {
		m := newPlanMem(rng, n, 1, 2)
		m.t.Phi = m.t.Phi[:n+d]
		p, err := m.plan()
		switch {
		case d > 0 && err == nil:
			t.Errorf("Phi longer than the other tables accepted")
		case d <= 0 && (err != nil || p.Rows() != n+d):
			t.Errorf("Phi at %+d: rows/err = %v", d, err)
		}
	}
	m := newPlanMem(rng, n, 1, 2)
	m.t = AllPassRows{}
	if _, err := m.plan(); err == nil || !strings.Contains(err.Error(), "no rows") {
		t.Errorf("zero rows: err = %v", err)
	}
}

// TestAllPass3PlanStrideBounds: a stride below n is refused, n accepted, above n
// accepted when the ring holds the slots.
func TestAllPass3PlanStrideBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(24))
	const n = 6
	for _, d := range []int{-1, 0, 1} {
		m := newPlanMem(rng, n, 1, 2)
		m.inStr = n + d
		m.in = append(m.in[:n:n], make([]complex128, 2)...) // room for the wider slot
		if _, err := m.plan(); (d < 0) != (err != nil) {
			t.Errorf("input stride %+d: err = %v", d, err)
		}
		for l := 0; l < 3; l++ {
			m := newPlanMem(rng, n, 1, 2)
			m.wStr[l] = n + d
			m.w[l] = append(m.w[l][:2*n:2*n], make([]complex128, 4)...)
			if _, err := m.plan(); (d < 0) != (err != nil) {
				t.Errorf("link %d stride %+d: err = %v", l, d, err)
			}
		}
	}
	// one slot of the ring short of a whole slot: the input ring needs inStride elements,
	// a link ring needs two whole slots
	m := newPlanMem(rng, n, 1, 2)
	m.inStr = n + 1
	m.in = m.in[:n]
	if _, err := m.plan(); err == nil {
		t.Errorf("input ring shorter than one stride accepted")
	}
	m = newPlanMem(rng, n, 1, 2)
	m.wStr[1] = n + 1
	m.w[1] = m.w[1][:2*n+1]
	if _, err := m.plan(); err == nil {
		t.Errorf("link ring shorter than two strides accepted")
	}
}

// overlaid builds the memory of a plan of n rows (one input slot, links of two slots) with
// every operand starting at the given float64 offset of one arena.
func overlaid(n int, start [12]int) *planMem {
	arena := make([]complex128, 40*n+80)
	base := unsafe.Pointer(&arena[0])
	cs := func(i, m int) []complex128 { return unsafe.Slice((*complex128)(unsafe.Add(base, 8*start[i])), m) }
	fs := func(i int) []float64 { return unsafe.Slice((*float64)(unsafe.Add(base, 8*start[i])), n) }
	return &planMem{
		t:   AllPassRows{Phi: cs(5, n), Q0: cs(6, n), Q1: cs(7, n), Q2: cs(8, n), C0: fs(9), C1: fs(10), C2: fs(11)},
		out: cs(0, n), in: cs(1, n),
		w:     [3][]complex128{cs(2, 2*n), cs(3, 2*n), cs(4, 2*n)},
		inStr: n, wStr: [3]int{n, n, n},
	}
}

var planNames = [12]string{"out", "in ring", "link 0", "link 1", "link 2", "Phi", "Q0", "Q1", "Q2", "C0", "C1", "C2"}

// TestAllPass3PlanOverlap: for every written memory (out, the link rings) and every other
// operand, an exact alias, an overlap of the last element and one of the first element
// are refused; the neighbour touching without sharing a byte, on either side, is
// accepted. Two read operands may overlap.
func TestAllPass3PlanOverlap(t *testing.T) {
	const n = 4
	size := func(i int) int { // bytes/8 of the whole operand
		switch {
		case i == 0 || i == 1 || (i >= 5 && i <= 8):
			return 2 * n
		case i >= 2 && i <= 4:
			return 4 * n
		}
		return n
	}
	elem := func(i int) int {
		if i >= 9 {
			return 1
		}
		return 2
	}
	var disjoint [12]int
	for i := range disjoint {
		disjoint[i] = (i + 1) * 8 * n
	}
	isW := func(i int) bool { return i <= 4 && i != 1 }
	for a := 0; a < 12; a++ {
		for b := 0; b < 12; b++ {
			if a == b || (!isW(a) && !isW(b)) {
				continue
			}
			for _, c := range []struct {
				name  string
				delta int
				over  bool
			}{
				{"alias", 0, true},
				{"last element of a", size(a) - elem(a), true},
				{"first element of a", -(size(b) - elem(b)), true},
				{"b ends where a starts", -size(b), false},
				{"b starts where a ends", size(a), false},
			} {
				start := disjoint
				start[b] = start[a] + c.delta
				_, err := overlaid(n, start).plan()
				if c.over != (err != nil) {
					t.Errorf("%s / %s (%s): err = %v, want refusal %v", planNames[a], planNames[b], c.name, err, c.over)
				}
			}
		}
	}
	if _, err := overlaid(n, disjoint).plan(); err != nil {
		t.Errorf("disjoint memory refused: %v", err)
	}
	// two read operands sharing memory: the tables with one another and with the ring
	s := disjoint
	s[6], s[7], s[1] = s[5], s[5], s[5]
	if _, err := overlaid(n, s).plan(); err != nil {
		t.Errorf("overlapping read-only operands refused: %v", err)
	}
}

func planPanics(fn func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	fn()
	return ""
}

// TestAllPass3PlanRunRefusals: the slot numbers are checked on both sides of each bound,
// nothing is written by a refused call, and a link may not read the slot it writes.
func TestAllPass3PlanRunRefusals(t *testing.T) {
	rng := rand.New(rand.NewSource(25))
	const n, inSlots, wSlots = 5, 3, 4
	m := newPlanMem(rng, n, inSlots, wSlots)
	p, err := m.plan()
	if err != nil {
		t.Fatal(err)
	}
	snap := func() [4][]complex128 {
		return [4][]complex128{cloneC(m.out), cloneC(m.w[0]), cloneC(m.w[1]), cloneC(m.w[2])}
	}
	ok := [3]int{0, 1, 2}
	okCur := [3]int{1, 2, 3}
	run := func(in int, o, c [3]int) string { return planPanics(func() { p.Run(in, o, c) }) }
	for _, c := range []struct {
		name string
		in   int
		o, c [3]int
		want string // "" = accepted
	}{
		{"last input slot", inSlots - 1, ok, okCur, ""},
		{"input slot past the end", inSlots, ok, okCur, "input slot"},
		{"negative input slot", -1, ok, okCur, "input slot"},
		{"last link slot read", 0, [3]int{wSlots - 1, 1, 2}, [3]int{0, 2, 3}, ""},
		{"last link slot written", 0, [3]int{0, 1, 2}, [3]int{wSlots - 1, 2, 3}, ""},
		{"read slot past the end, link 0", 0, [3]int{wSlots, 1, 2}, okCur, "link slot"},
		{"read slot past the end, link 1", 0, [3]int{0, wSlots, 2}, okCur, "link slot"},
		{"read slot past the end, link 2", 0, [3]int{0, 1, wSlots}, okCur, "link slot"},
		{"written slot past the end, link 0", 0, ok, [3]int{wSlots, 2, 3}, "link slot"},
		{"written slot past the end, link 1", 0, ok, [3]int{1, wSlots, 3}, "link slot"},
		{"written slot past the end, link 2", 0, ok, [3]int{1, 2, wSlots}, "link slot"},
		{"negative read slot", 0, [3]int{-1, 1, 2}, okCur, "link slot"},
		{"negative written slot", 0, ok, [3]int{1, -1, 3}, "link slot"},
		{"link 0 reads what it writes", 0, [3]int{2, 1, 2}, [3]int{2, 2, 3}, "reads the slot it writes"},
		{"link 1 reads what it writes", 0, [3]int{0, 3, 2}, [3]int{1, 3, 3}, "reads the slot it writes"},
		{"link 2 reads what it writes", 0, [3]int{0, 1, 0}, [3]int{1, 2, 0}, "reads the slot it writes"},
	} {
		before := snap()
		msg := run(c.in, c.o, c.c)
		if c.want == "" && msg != "" || c.want != "" && !strings.Contains(msg, c.want) {
			t.Errorf("%s: panic %q, want %q", c.name, msg, c.want)
		}
		if c.want != "" {
			after := snap()
			for i := range before {
				for k := range before[i] {
					if before[i][k] != after[i][k] {
						t.Fatalf("%s: a refused call wrote memory (%d, %d)", c.name, i, k)
					}
				}
			}
		}
	}
}

// TestAllPass3PlanCallDoesNotAllocate: a plan's call allocates nothing.
func TestAllPass3PlanCallDoesNotAllocate(t *testing.T) {
	rng := rand.New(rand.NewSource(26))
	m := newPlanMem(rng, 34, 3, 6)
	p, err := m.plan()
	if err != nil {
		t.Fatal(err)
	}
	if a := testing.AllocsPerRun(50, func() { p.Run(1, [3]int{0, 1, 2}, [3]int{1, 2, 3}) }); a != 0 {
		t.Errorf("Run allocates %v times", a)
	}
}

func BenchmarkAllPass3Plan(b *testing.B) {
	const n = 34
	rng := rand.New(rand.NewSource(1))
	m := newPlanMem(rng, n, 3, 6)
	p, err := m.plan()
	if err != nil {
		b.Fatal(err)
	}
	b.Run("plan", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.Run(1, [3]int{0, 1, 2}, [3]int{1, 2, 3})
		}
	})
	b.Run("checked", func(b *testing.B) {
		var old, cur [3][]complex128
		for l := range old {
			old[l], cur[l] = m.w[l][l*n:l*n+n], m.w[l][(l+1)*n:(l+1)*n+n]
		}
		for i := 0; i < b.N; i++ {
			AllPass3(m.out, m.in[n:2*n], &m.t, &old, &cur)
		}
	})
}

// TestSpansMeetTouching: spans that only touch have an OR of zero (not negative): not a
// meeting; one byte of overlap is.
func TestSpansMeetTouching(t *testing.T) {
	for _, c := range []struct {
		name  string
		w, r  []span
		meets bool
	}{
		{"read starts where written ends", []span{{0, 8}}, []span{{8, 16}}, false},
		{"read ends where written starts", []span{{8, 16}}, []span{{0, 8}}, false},
		{"written touch", []span{{0, 8}, {8, 16}}, nil, false},
		{"one byte shared", []span{{0, 8}}, []span{{7, 16}}, true},
		{"one byte shared, other side", []span{{7, 16}}, []span{{0, 8}}, true},
		{"written ones share a byte", []span{{0, 8}, {7, 16}}, nil, true},
		{"no written", nil, []span{{0, 8}, {0, 8}}, false},
	} {
		if got := spansMeet(c.w, c.r); got != c.meets {
			t.Errorf("%s: %v, want %v", c.name, got, c.meets)
		}
	}
}
