package floats

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

const (
	psmPad  = 16 // elements past maxEnd+7 in every backing array
	psmFill = float32(-7777.5)
	psmU    = 0x1p-24
)

// psmFix is one call of PSMixPlan32.Run: every plane is a slice of exactly maxEnd+7 elements
// over a backing array of psmPad more, whose tail holds a sentinel that no call may touch.
type psmFix struct {
	runs      []PSMixRun
	nBands    int
	plan      *PSMixPlan32
	gains, tg []float32
	s, l, r   Split32
	d         [3]Split32
	need      int
}

func psmSlice(n int, val func() float32) []float32 {
	b := make([]float32, n+psmPad)
	for i := range b {
		b[i] = psmFill
	}
	for i := 0; i < n; i++ {
		b[i] = val()
	}
	return b[:n]
}

// planes lists the operand planes that a call reads or writes, the written ones (l, r) first.
func (f *psmFix) planes() []*[]float32 {
	p := []*[]float32{&f.l.Re, &f.l.Im, &f.r.Re, &f.r.Im, &f.s.Re, &f.s.Im}
	for q := range f.d {
		p = append(p, &f.d[q].Re, &f.d[q].Im)
	}
	return p
}

// newPSMFix draws a fixture of runs run lengths 1..maxLen, a gap of 0 to 2 rows between
// runs, magnitudes spread over 10^-spread .. 10^spread; a source that no run uses is empty.
func newPSMFix(rng *rand.Rand, nBands, nRuns, maxLen int, spread float64) *psmFix {
	f := &psmFix{nBands: nBands}
	row := rng.Intn(3)
	for i := 0; i < nRuns; i++ {
		n := 1 + rng.Intn(maxLen)
		f.runs = append(f.runs, PSMixRun{row, row + n, rng.Intn(nBands), rng.Intn(3)})
		row += n + rng.Intn(3)
	}
	f.finish(rng, spread)
	return f
}

// finish builds the plan and the operands of f.runs.
func (f *psmFix) finish(rng *rand.Rand, spread float64) {
	var err error
	if f.plan, err = NewPSMixPlan32(f.runs, f.nBands); err != nil {
		panic(err)
	}
	f.need = f.plan.maxEnd + psMixSlack
	val := func() float32 {
		return float32(rng.NormFloat64() * math.Pow(10, spread*(2*rng.Float64()-1)))
	}
	f.gains = psmSlice(4*f.nBands, val)
	f.tg = psmSlice(f.nBands, func() float32 { return rng.Float32() })
	for i, p := range f.planes() {
		n := f.need
		if i >= 6 && f.plan.srcEnd[(i-6)/2] == 0 {
			*p = nil // a source no run uses
			continue
		}
		if i < 4 {
			*p = psmSlice(n, func() float32 { return psmFill })
		} else {
			*p = psmSlice(n, val)
		}
	}
}

func (f *psmFix) clone() *psmFix {
	g := *f
	g.gains = append([]float32(nil), f.gains...)
	g.tg = append([]float32(nil), f.tg...)
	src, dst := f.planes(), g.planes()
	for i := range src {
		if *src[i] == nil {
			*dst[i] = nil
			continue
		}
		full := (*src[i])[:len(*src[i])+psmPad]
		*dst[i] = append([]float32(nil), full...)[:len(*src[i])]
	}
	return &g
}

func (f *psmFix) call(fn func(*PSMixPlan32, []float32, []float32, Split32, *[3]Split32, Split32, Split32)) {
	fn(f.plan, f.gains, f.tg, f.s, &f.d, f.l, f.r)
}

func psmPublic(p *PSMixPlan32, gains, tg []float32, s Split32, d *[3]Split32, l, r Split32) {
	p.Run(gains, tg, s, d, l, r)
}

// padIntact reports the first plane whose padding past the slice was written.
func (f *psmFix) padIntact() string {
	for i, p := range f.planes() {
		if *p == nil {
			continue
		}
		full := (*p)[:len(*p)+psmPad]
		for k := len(*p); k < len(full); k++ {
			if full[k] != psmFill {
				return fmt.Sprintf("plane %d element %d past the %d required was written (%v)", i, k, len(*p), full[k])
			}
		}
	}
	return ""
}

func psmBig(x float32) *big.Float { return new(big.Float).SetPrec(200).SetFloat64(float64(x)) }

// psmCheck compares the rows of the runs of f (already run) with the 200-bit result, using the
// inputs of in0 (the same fixture before the call), and returns the worst ratio of an error to
// the documented bound.
func psmCheck(f, in0 *psmFix) (worst float64, msg string) {
	for _, run := range f.runs {
		g := in0.gains[4*run.Band : 4*run.Band+4]
		tg := psmBig(in0.tg[run.Band])
		for k := run.Start; k < run.End; k++ {
			for c := 0; c < 2; c++ {
				pick := func(sp Split32) float32 { return [2][]float32{sp.Re, sp.Im}[c][k] }
				s, d := psmBig(pick(in0.s)), psmBig(pick(in0.d[run.Src]))
				for side := 0; side < 2; side++ {
					gs, gd := psmBig(g[2*side]), new(big.Float).SetPrec(200).Mul(psmBig(g[2*side+1]), tg)
					a := new(big.Float).SetPrec(200).Mul(gs, s)
					b := new(big.Float).SetPrec(200).Mul(gd, d)
					exact := new(big.Float).SetPrec(200).Add(a, b)
					out := [2]Split32{f.l, f.r}[side]
					got := pick(out)
					diff := new(big.Float).SetPrec(200).Sub(psmBig(got), exact)
					e, _ := diff.Abs(diff).Float64()
					fa, _ := a.Abs(a).Float64()
					fb, _ := b.Abs(b).Float64()
					bound := 3 * psmU * (fa + fb) * (1 + 1e-6)
					if bound > 0 {
						worst = max(worst, e/bound)
					}
					if e > bound {
						return worst, fmt.Sprintf("run %v row %d side %d component %d: error %.3g exceeds the bound %.3g", run, k, side, c, e, bound)
					}
				}
			}
		}
	}
	return worst, ""
}

func psmCheckAll(t *testing.T, name string, f *psmFix, fn func(*PSMixPlan32, []float32, []float32, Split32, *[3]Split32, Split32, Split32)) float64 {
	t.Helper()
	in0 := f.clone()
	f.call(fn)
	if m := f.padIntact(); m != "" {
		t.Fatalf("%s: %s", name, m)
	}
	worst, msg := psmCheck(f, in0)
	if msg != "" {
		t.Fatalf("%s: %s", name, msg)
	}
	return worst
}

// TestPSMixPlan32Bound: the documented bound holds on every row of every run, for the public
// Run (the kernel where there is one) and for the portable loop, at run lengths from 1 to 20
// (every tail of the vector loops of both kernels), many runs per call, 1 to 40 bands; nothing
// past the required length is written.
func TestPSMixPlan32Bound(t *testing.T) {
	rng := rand.New(rand.NewSource(2027))
	var worst float64
	for rep := 0; rep < 400; rep++ {
		nb, nr, ml := 1+rng.Intn(40), 1+rng.Intn(40), 1+rng.Intn(20)
		f := newPSMFix(rng, nb, nr, ml, 2)
		g := f.clone()
		worst = max(worst, psmCheckAll(t, "Run", f, psmPublic), psmCheckAll(t, "portable", g, psMix32Ref))
	}
	t.Logf("worst error / bound = %.3f", worst)
	if worst < 0.01 {
		t.Errorf("worst error / bound = %.3g: the bound is not exercised", worst)
	}
}

// TestPSMixPlan32OneRun: every run length 1..40 alone, at every first row 0..9.
func TestPSMixPlan32OneRun(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for n := 1; n <= 40; n++ {
		for start := 0; start < 10; start++ {
			f := &psmFix{nBands: 3, runs: []PSMixRun{{start, start + n, rng.Intn(3), rng.Intn(3)}}}
			f.finish(rng, 1)
			psmCheckAll(t, fmt.Sprintf("n=%d start=%d", n, start), f, psmPublic)
		}
	}
}

// TestPSMixPlan32Overwrite: the rows past a run that a kernel computes are overwritten by the
// next run: adjacent runs of 1 row, each with its own band and source, all come out right.
func TestPSMixPlan32Overwrite(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	f := &psmFix{nBands: 30}
	for k := 0; k < 30; k++ {
		f.runs = append(f.runs, PSMixRun{k, k + 1, k, k % 3})
	}
	f.finish(rng, 1)
	psmCheckAll(t, "one-row runs", f, psmPublic)
}

// TestPSMixPlan32BoundDetects: the check is not vacuous: a result off by a few 1e-6 relative
// is refused.
func TestPSMixPlan32BoundDetects(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	f := &psmFix{nBands: 2, runs: []PSMixRun{{0, 5, 1, 0}}}
	f.finish(rng, 0)
	in0 := f.clone()
	f.call(psmPublic)
	if _, msg := psmCheck(f, in0); msg != "" {
		t.Fatal(msg)
	}
	f.l.Re[2] = f.l.Re[2]*(1+1e-5) + 1e-5
	if _, msg := psmCheck(f, in0); !strings.Contains(msg, "row 2 side 0 component 0") {
		t.Errorf("a result off by 1e-5 relative was accepted: %q", msg)
	}
}

func psmPanics(fn func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
		}
	}()
	fn()
	return ""
}

// TestNewPSMixPlan32Refuses: every way a run list can be wrong is an error.
func TestNewPSMixPlan32Refuses(t *testing.T) {
	ok := []PSMixRun{{0, 3, 0, 0}, {3, 4, 1, 2}}
	if _, err := NewPSMixPlan32(ok, 2); err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		runs []PSMixRun
		nb   int
	}{
		"no run":        {nil, 2},
		"no band":       {ok, 0},
		"empty run":     {[]PSMixRun{{2, 2, 0, 0}}, 2},
		"negative run":  {[]PSMixRun{{3, 2, 0, 0}}, 2},
		"huge row":      {[]PSMixRun{{0, 1 << 30, 0, 0}}, 2},
		"band too high": {[]PSMixRun{{0, 1, 2, 0}}, 2},
		"negative band": {[]PSMixRun{{0, 1, -1, 0}}, 2},
		"source 3":      {[]PSMixRun{{0, 1, 0, 3}}, 2},
		"negative src":  {[]PSMixRun{{0, 1, 0, -1}}, 2},
		"overlap":       {[]PSMixRun{{0, 3, 0, 0}, {2, 4, 0, 0}}, 2},
		"out of order":  {[]PSMixRun{{4, 5, 0, 0}, {0, 1, 0, 0}}, 2},
	} {
		if p, err := NewPSMixPlan32(c.runs, c.nb); err == nil || p != nil {
			t.Errorf("%s: accepted", name)
		}
	}
	p, _ := NewPSMixPlan32(ok, 2)
	if p.Runs() != 2 {
		t.Errorf("Runs() = %d", p.Runs())
	}
}

// TestPSMixPlan32Short: operands of exactly the required length are accepted, every one an
// element shorter panics, and a source no run uses may be empty.
func TestPSMixPlan32Short(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	mk := func() *psmFix {
		f := &psmFix{nBands: 2, runs: []PSMixRun{{1, 6, 0, 0}, {7, 9, 1, 2}}}
		f.finish(rng, 0)
		return f
	}
	f := mk()
	if msg := psmPanics(func() { f.call(psmPublic) }); msg != "" {
		t.Fatalf("exact lengths panicked: %s", msg)
	}
	for i, p := range mk().planes() {
		if *p == nil {
			continue
		}
		g := mk()
		q := g.planes()[i]
		if i >= 6 {
			*q = (*q)[:g.plan.srcEnd[(i-6)/2]+psMixSlack-1] // a source need only reach its last run
		} else {
			*q = (*q)[:len(*q)-1]
		}
		if msg := psmPanics(func() { g.call(psmPublic) }); !strings.Contains(msg, "shorter") {
			t.Errorf("plane %d one element short: panic %q", i, msg)
		}
	}
	for _, c := range []struct {
		name string
		cut  func(g *psmFix)
	}{{"gains", func(g *psmFix) { g.gains = g.gains[:7] }}, {"tg", func(g *psmFix) { g.tg = g.tg[:1] }}} {
		g := mk()
		c.cut(g)
		if msg := psmPanics(func() { g.call(psmPublic) }); !strings.Contains(msg, "shorter") {
			t.Errorf("%s one element short: panic %q", c.name, msg)
		}
	}
}

// ---- fuzz ----

// psmFromBytes decodes a fixture from fuzz input: per run a length byte (1 to 24), a gap
// byte, a band byte and a source byte, then 4 gains and a transient gain per band and 3
// float32 per row (s re/im shared, d re/im per run source); values outside 1e-4 .. 1e4 and
// non-finite ones are replaced by zero so that the relative bound is meaningful.
func psmFromBytes(data []byte) *psmFix {
	const nBands = 6
	if len(data) < 5 {
		return nil
	}
	nr := 1 + int(data[0])%12
	if len(data) < 1+4*nr {
		return nil
	}
	f := &psmFix{nBands: nBands}
	row := 0
	i := 1
	for r := 0; r < nr; r++ {
		n := 1 + int(data[i])%24
		f.runs = append(f.runs, PSMixRun{row, row + n, int(data[i+2]) % nBands, int(data[i+3]) % 3})
		row += n + int(data[i+1])%3
		i += 4
	}
	rest, k := data[i:], 0
	next := func() float32 {
		if 4*k+4 > len(rest) {
			return 0
		}
		v := math.Float32frombits(binary.LittleEndian.Uint32(rest[4*k:]))
		k++
		if a := math.Abs(float64(v)); !(a >= 1e-4 && a <= 1e4) {
			return 0
		}
		return v
	}
	var err error
	if f.plan, err = NewPSMixPlan32(f.runs, nBands); err != nil {
		panic(err)
	}
	f.need = f.plan.maxEnd + psMixSlack
	f.gains = psmSlice(4*nBands, next)
	f.tg = psmSlice(nBands, func() float32 { return float32(math.Abs(float64(next()))) / 1e4 })
	for j, p := range f.planes() {
		switch {
		case j >= 6 && f.plan.srcEnd[(j-6)/2] == 0:
			*p = nil
		case j < 4:
			*p = psmSlice(f.need, func() float32 { return psmFill })
		default:
			*p = psmSlice(f.need, next)
		}
	}
	return f
}

// FuzzPSMixPlan32 drives Run with runs and values carved from the fuzz input: the result must
// satisfy the documented bound and leave the padding alone.
func FuzzPSMixPlan32(f *testing.F) {
	rng := rand.New(rand.NewSource(1))
	for _, runs := range []int{1, 2, 3, 6, 12} {
		seed := []byte{byte(runs - 1)}
		for r := 0; r < runs; r++ {
			seed = append(seed, byte(rng.Intn(24)), byte(rng.Intn(3)), byte(rng.Intn(6)), byte(rng.Intn(3)))
		}
		for i := 0; i < 200; i++ {
			seed = binary.LittleEndian.AppendUint32(seed, math.Float32bits(float32(rng.NormFloat64())))
		}
		f.Add(seed)
	}
	f.Add([]byte{0, 0, 0, 0, 0})
	f.Add([]byte{1, 2, 3})
	f.Add([]byte{3, 1, 1, 1, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		x := psmFromBytes(data)
		if x == nil {
			return
		}
		psmCheckAll(t, "fuzz", x, psmPublic)
	})
}

// TestPSMixPlan32FromBytes: the decoder of the fuzz input refuses what is too short, takes at
// most 12 runs, and replaces non-finite values by zero.
func TestPSMixPlan32FromBytes(t *testing.T) {
	if psmFromBytes([]byte{1, 2, 3}) != nil || psmFromBytes([]byte{2, 0, 0, 0, 0}) != nil {
		t.Error("a short input gave a fixture")
	}
	nan := binary.LittleEndian.AppendUint32(nil, math.Float32bits(float32(math.NaN())))
	data := make([]byte, 1+4*12)
	data[0] = 11
	for i := 0; i < 400; i++ {
		data = append(data, nan...)
	}
	if x := psmFromBytes(data); x == nil || len(x.runs) != 12 || x.gains[0] != 0 {
		t.Errorf("non-finite input was not replaced by zero, or the run count is not capped at 12")
	}
}
