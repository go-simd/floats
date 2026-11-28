package floats

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

const (
	ap32Pad  = 16 // elements past n in every backing array
	ap32Fill = float32(-7777.5)
)

// ap32Fix is one call of AllPass3Split32: every operand is a slice of n elements over a
// backing array of n+ap32Pad whose tail holds a sentinel that no call may touch.
type ap32Fix struct {
	n        int
	out, in  Split32
	t        AllPassRows32
	old, cur [3]Split32
}

// ops lists the 27 operand slices in a fixed order; the first eight are the written ones
// (out.Re, out.Im, then cur[l].Re, cur[l].Im).
func (f *ap32Fix) ops() []*[]float32 {
	o := []*[]float32{&f.out.Re, &f.out.Im}
	for l := range f.cur {
		o = append(o, &f.cur[l].Re, &f.cur[l].Im)
	}
	o = append(o, &f.in.Re, &f.in.Im, &f.t.PhiRe, &f.t.PhiIm, &f.t.Q0Re, &f.t.Q0Im, &f.t.Q1Re, &f.t.Q1Im, &f.t.Q2Re, &f.t.Q2Im, &f.t.C0, &f.t.C1, &f.t.C2)
	for l := range f.old {
		o = append(o, &f.old[l].Re, &f.old[l].Im)
	}
	return o
}

// ap32Written is the number of written operands at the head of ops.
const ap32Written = 8

func ap32Slice(n int, vals func() float32) []float32 {
	b := make([]float32, n+ap32Pad)
	for i := range b {
		b[i] = ap32Fill
	}
	for i := 0; i < n; i++ {
		b[i] = vals()
	}
	return b[:n]
}

// newAP32Fix draws a fixture: magnitudes spread over 10^-spread .. 10^spread, C in [-1, 1].
func newAP32Fix(rng *rand.Rand, n int, spread float64) *ap32Fix {
	val := func() float32 {
		return float32(rng.NormFloat64() * math.Pow(10, spread*(2*rng.Float64()-1)))
	}
	coef := func() float32 { return float32(2*rng.Float64() - 1) }
	zero := func() float32 { return 0 }
	f := &ap32Fix{n: n}
	for _, p := range f.ops() {
		*p = ap32Slice(n, val)
	}
	for _, p := range []*[]float32{&f.t.C0, &f.t.C1, &f.t.C2} {
		*p = ap32Slice(n, coef)
	}
	// the written operands start as the sentinel only: a kernel that reads one is caught by
	// the comparison with the reference
	for _, p := range f.ops()[:ap32Written] {
		*p = ap32Slice(n, zero)
	}
	return f
}

// clone copies the values of the n elements and of the padding behind them.
func (f *ap32Fix) clone() *ap32Fix {
	g := &ap32Fix{n: f.n}
	src, dst := f.ops(), g.ops()
	for i := range src {
		full := (*src[i])[:len(*src[i])+ap32Pad]
		*dst[i] = append([]float32(nil), full...)[:len(*src[i])]
	}
	return g
}

func (f *ap32Fix) call(fn func(n int, out, in Split32, t *AllPassRows32, old, cur *[3]Split32)) {
	fn(f.n, f.out, f.in, &f.t, &f.old, &f.cur)
}

// padIntact reports the first operand whose padding past n was written.
func (f *ap32Fix) padIntact() string {
	for i, p := range f.ops() {
		full := (*p)[:len(*p)+ap32Pad]
		for k := f.n; k < len(full); k++ {
			if full[k] != ap32Fill {
				return fmt.Sprintf("operand %d element %d past n was written (%v)", i, k, full[k])
			}
		}
	}
	return ""
}

const ap32U = 0x1p-24

func bc32(re, im float32) bc { return bcOf(complex(float64(re), float64(im))) }

// ap32Bound checks row k of f (already run) against the 200-bit chain and the recurrence of
// the AllPass3Split32 documentation; in0 holds the inputs before the call. It returns the
// worst ratio of an error to its bound.
func ap32Bound(f, in0 *ap32Fix, k int) (worst float64, err string) {
	q := [3]bc{bc32(f.t.Q0Re[k], f.t.Q0Im[k]), bc32(f.t.Q1Re[k], f.t.Q1Im[k]), bc32(f.t.Q2Re[k], f.t.Q2Im[k])}
	cl := [3]float64{float64(f.t.C0[k]), float64(f.t.C1[k]), float64(f.t.C2[k])}
	phi, in := bc32(f.t.PhiRe[k], f.t.PhiIm[k]), bc32(in0.in.Re[k], in0.in.Im[k])
	x := phi.mul(in)
	ex := 3 * ap32U * phi.abs() * in.abs()
	check := func(name string, re, im float32, want bc, bound float64) string {
		e := bc32(re, im).sub(want).abs()
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
		old := bc32(in0.old[l].Re[k], in0.old[l].Im[k])
		p := q[l].mul(old)
		ep := 3 * ap32U * q[l].abs() * old.abs()
		c := x.add(p.scale(cl[l]))
		ac := math.Abs(cl[l])
		ec := ex + ac*ep + ap32U*(ac*p.abs()+c.abs())
		if m := check(fmt.Sprintf("cur[%d]", l), f.cur[l].Re[k], f.cur[l].Im[k], c, ec); m != "" {
			return worst, m
		}
		x = p.sub(c.scale(cl[l]))
		ex = ep + ac*ec + ap32U*(ac*c.abs()+x.abs())
	}
	return worst, check("out", f.out.Re[k], f.out.Im[k], x, ex)
}

// ap32CheckAll applies fn to a copy of f and checks every row against the bound and the
// padding against the sentinel.
func ap32CheckAll(t *testing.T, name string, f *ap32Fix, fn func(n int, out, in Split32, t *AllPassRows32, old, cur *[3]Split32)) float64 {
	t.Helper()
	in0 := f.clone()
	f.call(fn)
	if m := f.padIntact(); m != "" {
		t.Fatalf("%s n=%d: %s", name, f.n, m)
	}
	var worst float64
	for k := 0; k < f.n; k++ {
		w, msg := ap32Bound(f, in0, k)
		worst = max(worst, w)
		if msg != "" {
			t.Fatalf("%s n=%d: %s", name, f.n, msg)
		}
	}
	return worst
}

// TestAllPass3Split32Bound: the documented bound holds, for out and every state written,
// at every row count of the vector loop and its tail (1..70) and on larger ones, for the
// public function (the kernel where there is one) and for the portable loop; nothing past n
// is written.
func TestAllPass3Split32Bound(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	pub := func(n int, out, in Split32, tb *AllPassRows32, old, cur *[3]Split32) {
		AllPass3Split32(Split32{out.Re[:n], out.Im[:n]}, in, tb, old, cur)
	}
	var worst float64
	ns := []int{100, 257}
	for n := 1; n <= 70; n++ {
		ns = append(ns, n)
	}
	for _, n := range ns {
		for rep := 0; rep < 6; rep++ {
			f := newAP32Fix(rng, n, 2)
			g := f.clone()
			worst = max(worst, ap32CheckAll(t, "AllPass3Split32", f, pub), ap32CheckAll(t, "portable", g, allPass3Split32Ref))
		}
	}
	t.Logf("worst error / bound = %.3f", worst)
	if worst < 0.01 {
		t.Errorf("worst error / bound = %.3g: the bound is not exercised", worst)
	}
}

// TestAllPass3Split32BoundDetects: the bound check is not vacuous: a result off by a few
// 1e-6 relative (some dozen times the unit of the bound) is refused.
func TestAllPass3Split32BoundDetects(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	f := newAP32Fix(rng, 4, 0)
	in0 := f.clone()
	AllPass3Split32(f.out, f.in, &f.t, &f.old, &f.cur)
	if _, msg := ap32Bound(f, in0, 2); msg != "" {
		t.Fatal(msg)
	}
	f.out.Re[2] = f.out.Re[2]*(1+1e-5) + 1e-5
	if _, msg := ap32Bound(f, in0, 2); !strings.Contains(msg, "out row 2") {
		t.Errorf("a result off by 1e-5 relative was accepted: %q", msg)
	}
	f.cur[1].Im[3] += 1e-2 * (1 + float32(math.Abs(float64(f.cur[1].Im[3]))))
	if _, msg := ap32Bound(f, in0, 3); !strings.Contains(msg, "cur[1] row 3") {
		t.Errorf("a state off by 1e-2 was accepted: %q", msg)
	}
}

// TestAllPass3Split32Matches64: on well-scaled data the float32 result agrees with the
// float64 AllPass3 on the same inputs to a few 2^-24 of the chain's scale, the property the
// decoder relies on.
func TestAllPass3Split32Matches64(t *testing.T) {
	rng := rand.New(rand.NewSource(64))
	const n = 50
	f := newAP32Fix(rng, n, 0)
	z := func(re, im []float32) []complex128 {
		s := make([]complex128, n)
		for k := range s {
			s[k] = complex(float64(re[k]), float64(im[k]))
		}
		return s
	}
	tb := AllPassRows{Phi: z(f.t.PhiRe, f.t.PhiIm), Q0: z(f.t.Q0Re, f.t.Q0Im), Q1: z(f.t.Q1Re, f.t.Q1Im), Q2: z(f.t.Q2Re, f.t.Q2Im)}
	for k := 0; k < n; k++ {
		tb.C0, tb.C1, tb.C2 = append(tb.C0, float64(f.t.C0[k])), append(tb.C1, float64(f.t.C1[k])), append(tb.C2, float64(f.t.C2[k]))
	}
	old := [3][]complex128{z(f.old[0].Re, f.old[0].Im), z(f.old[1].Re, f.old[1].Im), z(f.old[2].Re, f.old[2].Im)}
	cur := [3][]complex128{make([]complex128, n), make([]complex128, n), make([]complex128, n)}
	out := make([]complex128, n)
	AllPass3(out, z(f.in.Re, f.in.Im), &tb, &old, &cur)
	f.call(allPass3Split32)
	var worst float64
	for k := 0; k < n; k++ {
		worst = max(worst, cmplxAbs(complex(float64(f.out.Re[k]), float64(f.out.Im[k]))-out[k]))
	}
	t.Logf("worst |float32 - float64| = %.3g (2^-24 = %.3g)", worst, ap32U)
	if worst > 256*ap32U {
		t.Errorf("float32 differs from float64 by %.3g", worst)
	}
}

func ap32Panics(fn func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
		}
	}()
	fn()
	return ""
}

// TestAllPass3Split32Empty: len(out.Re) = 0 reads and writes nothing, whatever the others are.
func TestAllPass3Split32Empty(t *testing.T) {
	if msg := ap32Panics(func() {
		AllPass3Split32(Split32{}, Split32{}, &AllPassRows32{}, &[3]Split32{}, &[3]Split32{})
	}); msg != "" {
		t.Errorf("n = 0 panicked: %s", msg)
	}
}

// TestAllPass3Split32Short: an operand of exactly n elements is accepted, one of n-1 panics
// (every one of the 26 operands besides out.Re, which defines n).
func TestAllPass3Split32Short(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for i := 1; i < 27; i++ {
		f := newAP32Fix(rng, 9, 0)
		if msg := ap32Panics(func() {
			f.call(func(n int, out, in Split32, tb *AllPassRows32, old, cur *[3]Split32) {
				AllPass3Split32(out, in, tb, old, cur)
			})
		}); msg != "" {
			t.Fatalf("exact lengths panicked: %s", msg)
		}
		g := newAP32Fix(rng, 9, 0)
		p := g.ops()[i]
		*p = (*p)[:8]
		msg := ap32Panics(func() { AllPass3Split32(g.out, g.in, &g.t, &g.old, &g.cur) })
		if !strings.Contains(msg, "shorter") {
			t.Errorf("operand %d one element short: panic %q", i, msg)
		}
	}
}

// TestAllPass3Split32Overlap: for every written operand and every other operand, the same
// memory and a one-element overlap panic, and a neighbour that only touches does not.
func TestAllPass3Split32Overlap(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const n = 5
	for w := 0; w < ap32Written; w++ {
		for o := 0; o < 27; o++ {
			if o == w {
				continue
			}
			for _, c := range []struct {
				name  string
				shift int
				over  bool
			}{{"same", 0, true}, {"one element", n - 1, true}, {"touching", n, false}} {
				f := newAP32Fix(rng, n, 0)
				ops := f.ops()
				base := (*ops[w])[:cap(*ops[w])]
				*ops[o] = base[c.shift : c.shift+n]
				msg := ap32Panics(func() { AllPass3Split32(f.out, f.in, &f.t, &f.old, &f.cur) })
				if c.over && !strings.Contains(msg, "overlaps") {
					t.Errorf("written %d / operand %d (%s): panic %q", w, o, c.name, msg)
				}
				if !c.over && msg != "" {
					t.Errorf("written %d / operand %d (%s): touching operands panicked: %s", w, o, c.name, msg)
				}
			}
		}
	}
}

// ---- fuzz ----

// ap32FromBytes decodes a fixture from fuzz input: 19 float32 per row (in 2, Phi 2, Q 6,
// old 6, C 3), values outside 1e-4 .. 1e4 and non-finite ones replaced by zero, so that no
// operation overflows or underflows and the relative bound is meaningful.
func ap32FromBytes(data []byte) *ap32Fix {
	const per = 19
	n := min(len(data)/(4*per), 24)
	if n == 0 {
		return nil
	}
	i := 0
	next := func() float32 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		i++
		if a := math.Abs(float64(v)); !(a >= 1e-4 && a <= 1e4) {
			return 0
		}
		return v
	}
	f := &ap32Fix{n: n}
	for _, p := range f.ops() {
		*p = ap32Slice(n, func() float32 { return 0 })
	}
	for k := 0; k < n; k++ {
		f.in.Re[k], f.in.Im[k] = next(), next()
		f.t.PhiRe[k], f.t.PhiIm[k] = next(), next()
		f.t.Q0Re[k], f.t.Q0Im[k], f.t.Q1Re[k], f.t.Q1Im[k], f.t.Q2Re[k], f.t.Q2Im[k] = next(), next(), next(), next(), next(), next()
		for l := range f.old {
			f.old[l].Re[k], f.old[l].Im[k] = next(), next()
		}
		f.t.C0[k], f.t.C1[k], f.t.C2[k] = next(), next(), next()
	}
	return f
}

// FuzzAllPass3Split32 drives AllPass3Split32 with rows carved from the fuzz input: the
// result must satisfy the documented bound and leave the padding alone.
func FuzzAllPass3Split32(f *testing.F) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 2, 3, 7, 9} {
		seed := make([]byte, 0, 4*19*n)
		for i := 0; i < 19*n; i++ {
			seed = binary.LittleEndian.AppendUint32(seed, math.Float32bits(float32(rng.NormFloat64())))
		}
		f.Add(seed)
	}
	f.Add([]byte{1, 2, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		x := ap32FromBytes(data)
		if x == nil {
			return
		}
		ap32CheckAll(t, "fuzz", x, func(n int, out, in Split32, tb *AllPassRows32, old, cur *[3]Split32) {
			AllPass3Split32(out, in, tb, old, cur)
		})
	})
}

// TestAllPass3Split32FromBytes: the decoder of the fuzz input takes the documented number of
// bytes per row and refuses what is too short.
func TestAllPass3Split32FromBytes(t *testing.T) {
	if ap32FromBytes(make([]byte, 4*19-1)) != nil {
		t.Error("a short input gave a fixture")
	}
	nan := binary.LittleEndian.AppendUint32(nil, math.Float32bits(float32(math.NaN())))
	data := make([]byte, 0, 4*19*30)
	for i := 0; i < 19*30; i++ {
		data = append(data, nan...)
	}
	if x := ap32FromBytes(data); x == nil || x.n != 24 || x.in.Re[0] != 0 {
		t.Errorf("non-finite input was not replaced by zero, or the row count is not capped at 24")
	}
}
