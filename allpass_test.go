package floats

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
	"unsafe"
)

var apSentinel = complex(7.5, -7.5)

// apFix is one call of AllPass3: every operand has tail elements past n holding a
// sentinel, so a kernel writing past n (or not writing within it) is seen.
type apFix struct {
	n        int
	out, in  []complex128
	t        AllPassRows
	old, cur [3][]complex128
}

// apBuf is a slice of n+tail elements starting off elements into a larger array, so the
// start is not always aligned to the vector width.
func apBuf(rng *rand.Rand, n, tail, off int, gen func() complex128) []complex128 {
	b := make([]complex128, off+n+tail)[off:]
	for i := range b {
		if i < n {
			b[i] = gen()
		} else {
			b[i] = apSentinel
		}
	}
	return b
}

func apFBuf(n, tail, off int, gen func() float64) []float64 {
	b := make([]float64, off+n+tail)[off:]
	for i := range b {
		if i < n {
			b[i] = gen()
		} else {
			b[i] = -7.5
		}
	}
	return b
}

// newAPFix draws the inputs: magnitudes spread over 2^-30 .. 2^30 so the products and sums
// carry real cancellation, coefficients in [-1.5, 1.5].
func newAPFix(rng *rand.Rand, n, tail, off int) *apFix {
	val := func() float64 { return rng.NormFloat64() * math.Ldexp(1, rng.Intn(61)-30) }
	cz := func() complex128 { return complex(val(), val()) }
	rot := func() complex128 { return complex(rng.NormFloat64(), rng.NormFloat64()) }
	cf := func() float64 { return rng.Float64()*3 - 1.5 }
	f := &apFix{n: n}
	f.out = apBuf(rng, n, tail, off, cz)
	f.in = apBuf(rng, n, tail, off, cz)
	f.t.Phi, f.t.Q0, f.t.Q1, f.t.Q2 = apBuf(rng, n, tail, off, rot), apBuf(rng, n, tail, off, rot), apBuf(rng, n, tail, off, rot), apBuf(rng, n, tail, off, rot)
	f.t.C0, f.t.C1, f.t.C2 = apFBuf(n, tail, off, cf), apFBuf(n, tail, off, cf), apFBuf(n, tail, off, cf)
	for l := range f.old {
		f.old[l] = apBuf(rng, n, tail, off, cz)
		f.cur[l] = apBuf(rng, n, tail, off, cz)
	}
	return f
}

func (f *apFix) clone() *apFix {
	g := *f
	g.out = append([]complex128(nil), f.out...)
	g.in = append([]complex128(nil), f.in...)
	g.t.Phi, g.t.Q0, g.t.Q1, g.t.Q2 = append([]complex128(nil), f.t.Phi...), append([]complex128(nil), f.t.Q0...), append([]complex128(nil), f.t.Q1...), append([]complex128(nil), f.t.Q2...)
	g.t.C0, g.t.C1, g.t.C2 = append([]float64(nil), f.t.C0...), append([]float64(nil), f.t.C1...), append([]float64(nil), f.t.C2...)
	for l := range f.old {
		g.old[l] = append([]complex128(nil), f.old[l]...)
		g.cur[l] = append([]complex128(nil), f.cur[l]...)
	}
	return &g
}

// diff returns a description of the first element where the written slices of f and g
// differ in bits, or "".
func (f *apFix) diff(g *apFix) string {
	same := func(a, b []complex128) int {
		for i := range a {
			if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) || math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
				return i
			}
		}
		return -1
	}
	if i := same(f.out, g.out); i >= 0 {
		return fmt.Sprintf("out[%d] = %v, want %v", i, f.out[i], g.out[i])
	}
	for l := range f.cur {
		if i := same(f.cur[l], g.cur[l]); i >= 0 {
			return fmt.Sprintf("cur[%d][%d] = %v, want %v", l, i, f.cur[l][i], g.cur[l][i])
		}
	}
	return ""
}

func (f *apFix) call(fn func(n int, out, in []complex128, t *AllPassRows, old, cur *[3][]complex128)) {
	fn(f.n, f.out, f.in, &f.t, &f.old, &f.cur)
}

// TestAllPass3Exact: every row count 1..70 (the pair loop, the odd tail, both together)
// at four starting offsets: the public function equals the portable loop in every bit of
// every operand, the sentinel tails included.
func TestAllPass3Exact(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for n := 1; n <= 70; n++ {
		for off := 0; off < 4; off++ {
			f := newAPFix(rng, n, 3, off)
			g := f.clone()
			AllPass3(f.out[:n], f.in, &f.t, &f.old, &f.cur)
			g.call(allPass3Ref)
			if d := f.diff(g); d != "" {
				t.Fatalf("n=%d off=%d: %s", n, off, d)
			}
			if f.out[n] != apSentinel || f.cur[2][n+2] != apSentinel {
				t.Fatalf("n=%d off=%d: a sentinel past n was overwritten", n, off)
			}
		}
	}
}

// TestAllPass3EmptyAndRowCount: len(out) = 0 reads and writes nothing (nil operands do
// not panic); the row count is len(out), not that of the operands.
func TestAllPass3EmptyAndRowCount(t *testing.T) {
	if msg := apPanics(func() { AllPass3(nil, nil, &AllPassRows{}, &[3][]complex128{}, &[3][]complex128{}) }); msg != "" {
		t.Errorf("len(out) = 0 with nil operands panicked: %s", msg)
	}
	rng := rand.New(rand.NewSource(3))
	f := newAPFix(rng, 9, 4, 0)
	z := f.clone()
	if msg := apPanics(func() { AllPass3(f.out[:0], f.in, &f.t, &f.old, &f.cur) }); msg != "" {
		t.Errorf("len(out) = 0 with operands panicked: %s", msg)
	}
	if d := f.diff(z); d != "" {
		t.Errorf("len(out) = 0 wrote %s", d)
	}
	g := f.clone()
	before := f.out[5]
	AllPass3(f.out[:5], f.in, &f.t, &f.old, &f.cur)
	allPass3Ref(5, g.out, g.in, &g.t, &g.old, &g.cur)
	if d := f.diff(g); d != "" {
		t.Fatal(d)
	}
	if f.out[5] != before {
		t.Fatal("row 5 was computed for len(out) = 5")
	}
}

// TestAllPass3Stack: the kernel keeps no pointer, a caller's rows stay on its stack.
func TestAllPass3Stack(t *testing.T) {
	allocs := testing.AllocsPerRun(20, func() {
		var out, in, phi, q0, q1, q2, o0, o1, o2, c0, c1, c2 [9]complex128
		var k0, k1, k2 [9]float64
		tab := AllPassRows{Phi: phi[:], Q0: q0[:], Q1: q1[:], Q2: q2[:], C0: k0[:], C1: k1[:], C2: k2[:]}
		old := [3][]complex128{o0[:], o1[:], o2[:]}
		cur := [3][]complex128{c0[:], c1[:], c2[:]}
		AllPass3(out[:], in[:], &tab, &old, &cur)
	})
	if allocs != 0 {
		t.Errorf("%v heap allocations per call", allocs)
	}
}

// operands names the sixteen operands of a fixture in the order of apOperand.
var apNames = []string{"out", "in", "Phi", "Q0", "Q1", "Q2", "C0", "C1", "C2", "old0", "old1", "old2", "cur0", "cur1", "cur2"}

// apShorten cuts operand i of f to m elements.
func apShorten(f *apFix, i, m int) {
	c := func(s []complex128) []complex128 { return s[:m] }
	fl := func(s []float64) []float64 { return s[:m] }
	switch i {
	case 0:
		f.out = c(f.out)
	case 1:
		f.in = c(f.in)
	case 2:
		f.t.Phi = c(f.t.Phi)
	case 3:
		f.t.Q0 = c(f.t.Q0)
	case 4:
		f.t.Q1 = c(f.t.Q1)
	case 5:
		f.t.Q2 = c(f.t.Q2)
	case 6:
		f.t.C0 = fl(f.t.C0)
	case 7:
		f.t.C1 = fl(f.t.C1)
	case 8:
		f.t.C2 = fl(f.t.C2)
	case 9, 10, 11:
		f.old[i-9] = c(f.old[i-9])
	default:
		f.cur[i-12] = c(f.cur[i-12])
	}
}

func apPanics(fn func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
		}
	}()
	fn()
	return ""
}

// TestAllPass3Short: an operand of exactly n elements is accepted, one of n-1 panics,
// for each of the fifteen operands in turn (out is the row count itself, so only the
// others can be short).
func TestAllPass3Short(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for i := 1; i < len(apNames); i++ {
		for _, n := range []int{1, 2, 5} {
			f := newAPFix(rng, n, 2, 0)
			apShorten(f, i, n)
			if msg := apPanics(func() { AllPass3(f.out[:n], f.in, &f.t, &f.old, &f.cur) }); msg != "" {
				t.Errorf("%s of exactly %d elements panicked: %s", apNames[i], n, msg)
			}
			g := newAPFix(rng, n, 2, 0)
			apShorten(g, i, n-1)
			msg := apPanics(func() { AllPass3(g.out[:n], g.in, &g.t, &g.old, &g.cur) })
			if !strings.Contains(msg, "shorter") {
				t.Errorf("%s of %d elements for n=%d: panic %q, want a shorter-operand panic", apNames[i], n-1, n, msg)
			}
		}
	}
}

// apOverlayed lays the fifteen operands of a fixture over one arena at the offsets
// start[i] (in float64 units), so that any two can be made to overlap or touch.
func apOverlayed(n int, start [15]int) *apFix {
	arena := make([]complex128, 80*n+80)
	base := unsafe.Pointer(&arena[0])
	cs := func(i int) []complex128 { return unsafe.Slice((*complex128)(unsafe.Add(base, 8*start[i])), n) }
	fs := func(i int) []float64 { return unsafe.Slice((*float64)(unsafe.Add(base, 8*start[i])), n) }
	f := &apFix{n: n}
	f.out, f.in = cs(0), cs(1)
	f.t = AllPassRows{Phi: cs(2), Q0: cs(3), Q1: cs(4), Q2: cs(5), C0: fs(6), C1: fs(7), C2: fs(8)}
	f.old = [3][]complex128{cs(9), cs(10), cs(11)}
	f.cur = [3][]complex128{cs(12), cs(13), cs(14)}
	return f
}

// TestAllPass3Overlap: for every written operand (out, cur_l) and every other operand,
// an exact alias, an overlap of the last element, and an overlap of the first element
// panic; the neighbour touching without sharing a byte, on either side, does not. The
// pairs of two read operands may overlap.
func TestAllPass3Overlap(t *testing.T) {
	const n = 4
	elem := func(i int) int { // size of an element of operand i, in float64 units
		if i >= 6 && i <= 8 {
			return 1
		}
		return 2
	}
	var disjoint [15]int
	for i := range disjoint {
		disjoint[i] = (i + 1) * 8 * n // far apart: a span is at most 2n units, a shift at most 2n
	}
	for a := 0; a < 15; a++ {
		for b := 0; b < 15; b++ {
			if a == b {
				continue
			}
			isW := func(i int) bool { return i == 0 || i >= 12 }
			if !isW(a) && !isW(b) {
				continue
			}
			spanA, spanB := n*elem(a), n*elem(b) // float64 units
			for _, c := range []struct {
				name  string
				delta int // start of b relative to a, float64 units
				over  bool
			}{
				{"alias", 0, true},
				{"last element of a", spanA - elem(a), true},
				{"first element of a", -(spanB - elem(b)), true},
				{"b ends where a starts", -spanB, false},
				{"b starts where a ends", spanA, false},
			} {
				start := disjoint
				start[b] = start[a] + c.delta
				f := apOverlayed(n, start)
				msg := apPanics(func() { AllPass3(f.out, f.in, &f.t, &f.old, &f.cur) })
				if c.over && !strings.Contains(msg, "overlaps") {
					t.Errorf("%s / %s (%s): panic %q, want an overlap panic", apNames[a], apNames[b], c.name, msg)
				}
				if !c.over && msg != "" {
					t.Errorf("%s / %s (%s): touching operands panicked: %s", apNames[a], apNames[b], c.name, msg)
				}
			}
		}
	}
}

// ---- the error bound against a 200-bit computation ----

type bc struct{ re, im *big.Float }

const apPrec = 200

func bcOf(z complex128) bc {
	return bc{new(big.Float).SetPrec(apPrec).SetFloat64(real(z)), new(big.Float).SetPrec(apPrec).SetFloat64(imag(z))}
}

func bnew() *big.Float { return new(big.Float).SetPrec(apPrec) }

func (a bc) mul(d bc) bc {
	return bc{bnew().Sub(bnew().Mul(a.re, d.re), bnew().Mul(a.im, d.im)), bnew().Add(bnew().Mul(a.re, d.im), bnew().Mul(a.im, d.re))}
}

func (a bc) add(b bc) bc { return bc{bnew().Add(a.re, b.re), bnew().Add(a.im, b.im)} }
func (a bc) sub(b bc) bc { return bc{bnew().Sub(a.re, b.re), bnew().Sub(a.im, b.im)} }
func (a bc) scale(c float64) bc {
	f := bnew().SetFloat64(c)
	return bc{bnew().Mul(a.re, f), bnew().Mul(a.im, f)}
}

// abs is the modulus, as a float64 (the bound only needs it to a few digits).
func (a bc) abs() float64 {
	s := bnew().Add(bnew().Mul(a.re, a.re), bnew().Mul(a.im, a.im))
	r, _ := bnew().Sqrt(s).Float64()
	return r
}

const apU = 0x1p-53

// apBound checks row k of f (already run) against the 200-bit chain and the recurrence of
// the AllPass3 documentation; it returns the worst ratio of an error to its bound.
func apBound(f *apFix, in0 *apFix, k int) (worst float64, err string) {
	q := [3]complex128{f.t.Q0[k], f.t.Q1[k], f.t.Q2[k]}
	cl := [3]float64{f.t.C0[k], f.t.C1[k], f.t.C2[k]}
	x := bcOf(f.t.Phi[k]).mul(bcOf(in0.in[k]))
	ex := 3 * apU * bcOf(f.t.Phi[k]).abs() * bcOf(in0.in[k]).abs()
	check := func(name string, got complex128, want bc, bound float64) string {
		e := bcOf(got).sub(want).abs()
		bound *= 1 + 1e-6
		if bound > 0 {
			worst = max(worst, e/bound)
		}
		if e > bound {
			return fmt.Sprintf("%s row %d: error %.3g exceeds the bound %.3g (|exact| = %.3g)", name, k, e, bound, want.abs())
		}
		return ""
	}
	for l := 0; l < 3; l++ {
		oldl := bcOf(in0.old[l][k])
		p := bcOf(q[l]).mul(oldl)
		ep := 3 * apU * bcOf(q[l]).abs() * oldl.abs()
		c := x.add(p.scale(cl[l]))
		ec := ex + math.Abs(cl[l])*ep + apU*c.abs()
		if m := check(fmt.Sprintf("cur[%d]", l), f.cur[l][k], c, ec); m != "" {
			return worst, m
		}
		x = p.sub(c.scale(cl[l]))
		ex = ep + math.Abs(cl[l])*ec + apU*x.abs()
	}
	return worst, check("out", f.out[k], x, ex)
}

// TestAllPass3Bound: the documented bound holds, for out and every state written, on
// random rows of widely spread magnitudes, for the public function (the kernel where
// there is one) and for the portable loop.
func TestAllPass3Bound(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	var worst float64
	for _, n := range []int{1, 2, 3, 8, 17, 34, 50} {
		for rep := 0; rep < 40; rep++ {
			f := newAPFix(rng, n, 2, 0)
			in0 := f.clone()
			g := f.clone()
			AllPass3(f.out[:n], f.in, &f.t, &f.old, &f.cur)
			g.call(allPass3Ref)
			for k := 0; k < n; k++ {
				for _, h := range []*apFix{f, g} {
					w, msg := apBound(h, in0, k)
					worst = max(worst, w)
					if msg != "" {
						t.Fatalf("n=%d: %s", n, msg)
					}
				}
			}
		}
	}
	t.Logf("worst error / bound = %.3f", worst)
	if worst < 0.01 {
		t.Errorf("worst error / bound = %.3g: the bound is not exercised", worst)
	}
}

// TestAllPass3BoundDetects: the bound check is not vacuous: one ulp of the result moved
// to a value 1e6 ulps off is refused.
func TestAllPass3BoundDetects(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	f := newAPFix(rng, 4, 0, 0)
	in0 := f.clone()
	AllPass3(f.out, f.in, &f.t, &f.old, &f.cur)
	if _, msg := apBound(f, in0, 2); msg != "" {
		t.Fatal(msg)
	}
	f.out[2] = complex(real(f.out[2])*(1+1e-10)+1e-10, imag(f.out[2]))
	if _, msg := apBound(f, in0, 2); !strings.Contains(msg, "out row 2") {
		t.Errorf("a result off by 1e-10 relative was accepted: %q", msg)
	}
	in0 = f.clone()
	f.cur[1][3] += complex(1e-3*(1+cmplxAbs(f.cur[1][3])), 0)
	if _, msg := apBound(f, in0, 3); !strings.Contains(msg, "cur[1] row 3") {
		t.Errorf("a state off by 1e-3 was accepted: %q", msg)
	}
}

func cmplxAbs(z complex128) float64 { return math.Hypot(real(z), imag(z)) }

// ---- fuzz ----

// apFromBytes decodes a fixture from fuzz input: 19 float64 per row (in 2, Phi 2, Q 6,
// old 6, C 3), values outside 1e-30 .. 1e30 and non-finite ones replaced by zero, so
// that no operation overflows or underflows and the relative bound is meaningful.
func apFromBytes(data []byte) *apFix {
	const per = 19
	n := min(len(data)/(8*per), 24)
	if n == 0 {
		return nil
	}
	i := 0
	next := func() float64 {
		v := math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
		i++
		if a := math.Abs(v); !(a >= 1e-30 && a <= 1e30) {
			return 0
		}
		return v
	}
	f := &apFix{n: n}
	zs := func() []complex128 {
		s := make([]complex128, n)
		for k := range s {
			s[k] = complex(next(), next())
		}
		return s
	}
	fs := func() []float64 { return make([]float64, n) }
	f.out, f.in = make([]complex128, n), nil
	f.in = zs()
	f.t.Phi, f.t.Q0, f.t.Q1, f.t.Q2 = zs(), zs(), zs(), zs()
	f.t.C0, f.t.C1, f.t.C2 = fs(), fs(), fs()
	for l := range f.old {
		f.old[l] = zs()
		f.cur[l] = make([]complex128, n)
	}
	for k := 0; k < n; k++ {
		f.t.C0[k], f.t.C1[k], f.t.C2[k] = next(), next(), next()
	}
	return f
}

// FuzzAllPass3 drives AllPass3 with rows carved from the fuzz input: the result must
// equal the portable loop bit for bit and satisfy the documented bound.
func FuzzAllPass3(f *testing.F) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 2, 3, 7} {
		seed := make([]byte, 0, 8*19*n)
		for i := 0; i < 19*n; i++ {
			seed = binary.LittleEndian.AppendUint64(seed, math.Float64bits(rng.NormFloat64()))
		}
		f.Add(seed)
	}
	f.Add([]byte{1, 2, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		x := apFromBytes(data)
		if x == nil {
			return
		}
		in0, g := x.clone(), x.clone()
		AllPass3(x.out, x.in, &x.t, &x.old, &x.cur)
		g.call(allPass3Ref)
		if d := x.diff(g); d != "" {
			t.Fatalf("n=%d: kernel and reference differ: %s", x.n, d)
		}
		for k := 0; k < x.n; k++ {
			if _, msg := apBound(x, in0, k); msg != "" {
				t.Fatal(msg)
			}
		}
	})
}

// TestAllPass3FromBytes: the decoder of the fuzz input takes the documented number of
// values per row and zeroes the ones it must.
func TestAllPass3FromBytes(t *testing.T) {
	if apFromBytes(make([]byte, 8*19-1)) != nil {
		t.Error("a short input made rows")
	}
	data := make([]byte, 8*19*30)
	for i := 0; i < len(data)/8; i++ {
		v := 1.0
		switch i {
		case 0:
			v = math.NaN()
		case 1:
			v = 1e31
		case 2:
			v = 1e-31
		case 3:
			v = 1e30
		case 4:
			v = -1e-30
		}
		binary.LittleEndian.PutUint64(data[8*i:], math.Float64bits(v))
	}
	f := apFromBytes(data)
	if f.n != 24 {
		t.Fatalf("n = %d, want 24 (capped)", f.n)
	}
	// values 0 (NaN) and 1 (1e31) are zeroed, 2 (1e-31) too, 3 (1e30) and 4 (-1e-30) are kept
	if f.in[0] != 0 || f.in[1] != complex(0, 1e30) || f.in[2] != complex(-1e-30, 1) {
		t.Errorf("in = %v, want [(0+0i) (0+1e+30i) (-1e-30+1i)]", f.in[:3])
	}
}
