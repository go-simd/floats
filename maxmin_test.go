package floats

import (
	"math"
	"math/rand"
	"testing"
)

// The oracle of every kernel is ndarray's scalar definition (go-ndarray/ndarray
// v0.44.0, internal/kernels Maximum / Minimum: dst[i] = max(a[i], b[i]) with the
// Go builtin), spelled out here by the rule the language gives it, which
// runtime/minmax.go (fmax / fmin) implements, and written independently of
// maximumRef: a NaN b wins, else a NaN a wins, else the larger (smaller); two zeros
// give the AND (OR) of their bits. Each kernel must agree with it bit for bit on
// every length, alignment and special value, and must not write past dst. Where
// the builtin returns a number the oracle is also held to it (TestMaxMinBuiltin).

type maxminOp struct {
	name string
	f    func(dst, a, b []float64)
	ref  func(dst, a, b []float64) // the fallback: the builtin loop
	rule func(dst, a, b []float64) // the rule loop that finishes kernel calls
	want func(x, y float64) float64
	bi   func(x, y float64) float64 // the Go builtin, ndarray's expression
}

func wantMax(x, y float64) float64 {
	xb, yb := math.Float64bits(x), math.Float64bits(y)
	switch {
	case y != y:
		return y
	case x != x:
		return x
	case x < y:
		return y
	case x > y:
		return x
	case x == 0: // both zeros
		return math.Float64frombits(xb & yb)
	}
	return x // equal numbers: same bits
}

func wantMin(x, y float64) float64 {
	xb, yb := math.Float64bits(x), math.Float64bits(y)
	switch {
	case y != y:
		return y
	case x != x:
		return x
	case x > y:
		return y
	case x < y:
		return x
	case x == 0:
		return math.Float64frombits(xb | yb)
	}
	return x
}

var maxminOps = []maxminOp{
	{"Maximum", Maximum, maximumRef, maximumRule, wantMax, func(x, y float64) float64 { return max(x, y) }},
	{"Minimum", Minimum, minimumRef, minimumRule, wantMin, func(x, y float64) float64 { return min(x, y) }},
}

// pairwiseFallback is set by the gate-off tests: with no kernel running, Maximum
// and Minimum are the builtin loop, so the expected bits are the builtin's.
var pairwiseFallback bool

func checkMaxMin(t *testing.T, op maxminOp, what string, got, a, b []float64, n, off int) {
	t.Helper()
	for i := 0; i < n; i++ {
		want := op.want(a[i], b[i])
		if pairwiseFallback {
			// The builtin's NaN is the target's and may depend on the call site
			// (the compiler may swap the operands): held to bits for a number, to
			// "is a NaN" for a NaN.
			if bw := op.bi(a[i], b[i]); !math.IsNaN(bw) || !math.IsNaN(got[i]) {
				want = bw
			} else {
				want = got[i]
			}
		}
		if math.Float64bits(got[i]) != math.Float64bits(want) {
			t.Fatalf("%s %s n=%d off=%d: [%d] %s(%v (%#x), %v (%#x)) = %v (%#x), want %v (%#x)", op.name, what, n, off, i,
				op.name, a[i], math.Float64bits(a[i]), b[i], math.Float64bits(b[i]),
				got[i], math.Float64bits(got[i]), want, math.Float64bits(want))
		}
	}
}

// maxminEdges: signed zeros, infinities, NaNs of every kind (quiet and
// signalling, both signs, with payloads), subnormals, and ordinary numbers
// including a pair of equal values and neighbours.
func maxminEdges() []float64 {
	var e []float64
	add := func(v float64) { e = append(e, v, -v) }
	for _, v := range []float64{
		0, math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64, 1e-310,
		0x1p-1022 - 0x1p-1074, 0x1p-1022, 1e-300, 0.5, math.Nextafter(1, 0), 1,
		math.Nextafter(1, 2), 2, 3, 1e100, math.MaxFloat64, math.Inf(1),
	} {
		add(v)
	}
	e = append(e, math.NaN(), negNaN, sigNaN, nsigNaN,
		math.Float64frombits(0x7ff8000000000001), math.Float64frombits(0xfff8000000000001),
		math.Float64frombits(0x7fffffffffffffff), math.Float64frombits(0xffffffffffffffff),
		math.Float64frombits(0x7ff7ffffffffffff), math.Float64frombits(0xfff7ffffffffffff),
		math.Float64frombits(0x7ff0000000000002), math.Float64frombits(0x7ff4000000000000))
	return e
}

// Every pair of edge values at every position of every length 0..33 and four
// alignments: a[i] = edges[(i+s) % len], b[i] = edges[(i+t) % len] for every s, t.
func TestPairwiseEdgeTable(t *testing.T) {
	edges := maxminEdges()
	rng := rand.New(rand.NewSource(1))
	for _, op := range maxminOps {
		for n := 0; n <= 33; n++ {
			for off := 0; off < 4; off++ {
				for s := range edges {
					for u := range edges {
						a, _ := fillE(n, off, rng, false)
						b, _ := fillE(n, (off+1)%4, rng, false)
						for i := 0; i < n; i++ {
							a[i] = edges[(i+s)%len(edges)]
							b[i] = edges[(i+u)%len(edges)]
						}
						dst, buf := fillE(n, (off+2)%4, rng, false)
						op.f(dst, a, b)
						checkMaxMin(t, op, "table", dst, a, b, n, off)
						checkGuard(t, op.name, buf, n, (off+2)%4)
					}
				}
			}
		}
	}
}

// pairWords returns a and b of n float64s: random bit patterns, with the pairs a
// kernel must decide on a tie or a zero planted often (b = a, b = -a, zeros, NaNs
// with random payloads, adjacent values).
func pairWords(n, off int, rng *rand.Rand) (a, b []float64) {
	a, _ = fillE(n, off, rng, false)
	b, _ = fillE(n, (off+1)%4, rng, false)
	nan := func() float64 {
		return math.Float64frombits(rng.Uint64()&^(0x7ff<<52) | 0x7ff<<52 | 1<<uint(rng.Intn(52)))
	}
	for i := range a {
		switch rng.Intn(9) {
		case 0:
			a[i], b[i] = math.Float64frombits(rng.Uint64()), math.Float64frombits(rng.Uint64())
		case 1:
			b[i] = a[i]
		case 2:
			b[i] = -a[i]
		case 3:
			a[i], b[i] = math.Copysign(0, float64(rng.Intn(2)*2-1)), math.Copysign(0, float64(rng.Intn(2)*2-1))
		case 4:
			a[i] = nan()
		case 5:
			b[i] = nan()
		case 6:
			a[i], b[i] = nan(), nan()
		case 7:
			b[i] = math.Nextafter(a[i], math.Inf(rng.Intn(2)*2-1))
		}
	}
	return a, b
}

func TestPairwiseRandomDifferential(t *testing.T) {
	for _, op := range maxminOps {
		forEachCase(t, 81, func(n, off int, sp bool, rng *rand.Rand) {
			a, b := pairWords(n, off, rng)
			dst, buf := fillE(n, (off+2)%4, rng, sp)
			op.f(dst, a, b)
			checkMaxMin(t, op, "random", dst, a, b, n, off)
			checkGuard(t, op.name, buf, n, (off+2)%4)
			op.rule(dst, a, b)
			saved := pairwiseFallback
			pairwiseFallback = false // the rule loop is held to the rule, not the builtin
			checkMaxMin(t, op, "rule", dst, a, b, n, off)
			pairwiseFallback = saved
			// The fallback is the builtin loop, bit for bit.
			op.ref(dst, a, b)
			for i := 0; i < n; i++ {
				if w := op.bi(a[i], b[i]); math.Float64bits(dst[i]) != math.Float64bits(w) && !(math.IsNaN(w) && math.IsNaN(dst[i])) {
					t.Fatalf("%s ref n=%d: [%d] = %#x, builtin %#x", op.name, n, i, math.Float64bits(dst[i]), math.Float64bits(w))
				}
			}
		})
	}
}

// dst may be a, or b, or both may be the same slice.
func TestPairwiseAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(82))
	for _, op := range maxminOps {
		for n := 0; n <= 70; n++ {
			a, b := pairWords(n, 1, rng)
			ca := append([]float64(nil), a...)
			op.f(ca, ca, b)
			checkMaxMin(t, op, "dst=a", ca, a, b, n, 0)
			cb := append([]float64(nil), b...)
			op.f(cb, a, cb)
			checkMaxMin(t, op, "dst=b", cb, a, b, n, 0)
			cc := append([]float64(nil), a...)
			op.f(cc, cc, cc)
			checkMaxMin(t, op, "a=b=dst", cc, a, a, n, 0)
			d := make([]float64, n)
			op.f(d, a, a)
			checkMaxMin(t, op, "a=b", d, a, a, n, 0)
		}
	}
}

// Only the first len(a) elements are written, and a short b or dst panics.
func TestPairwiseLengths(t *testing.T) {
	a := []float64{1, -2, 3, -0.5, 0.5, 8, -9, 10, 11, -12}
	b := []float64{2, -1, 3, 0.5, -0.5, 7, -8, 11, 10, -13, 5, 6}
	for _, op := range maxminOps {
		dst := make([]float64, len(a)+3)
		for i := range dst {
			dst[i] = 99
		}
		op.f(dst, a[:9], b)
		for i := 0; i < 9; i++ {
			if want := op.want(a[i], b[i]); math.Float64bits(dst[i]) != math.Float64bits(want) {
				t.Fatalf("%s: [%d] = %v, want %v", op.name, i, dst[i], want)
			}
		}
		if dst[9] != 99 || dst[10] != 99 || dst[11] != 99 {
			t.Fatalf("%s touched dst past len(a): %v", op.name, dst)
		}
		for _, bad := range []func(){
			func() { op.f(dst[:len(a)-1], a, b) },
			func() { op.f(dst, a, b[:len(a)-1]) },
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Fatalf("%s did not panic on a short slice", op.name)
					}
				}()
				bad()
			}()
		}
	}
}

// The documented values, by hand, in both operand orders.
func TestPairwiseDocumentedValues(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	type tc struct{ a, b, max, min float64 }
	cases := []tc{
		{1, 2, 2, 1}, {-1, -2, -1, -2}, {inf, 1, inf, 1}, {-inf, 1, 1, -inf},
		{negZero, 0, 0, negZero}, {0, negZero, 0, negZero},
		{negZero, negZero, negZero, negZero}, {0, 0, 0, 0},
		{nan, inf, nan, nan}, {inf, nan, nan, nan}, {nan, -inf, nan, nan},
		{nsigNaN, 1, nsigNaN, nsigNaN}, {1, sigNaN, sigNaN, sigNaN},
		{negNaN, nan, nan, nan}, {nan, negNaN, negNaN, negNaN}, // two NaNs: b
	}
	pad := func(v float64) []float64 {
		s := make([]float64, 16)
		for i := range s {
			s[i] = v
		}
		return s
	}
	for _, c := range cases {
		for _, op := range maxminOps {
			want := c.max
			if op.name == "Minimum" {
				want = c.min
			}
			got := make([]float64, 16)
			op.f(got, pad(c.a), pad(c.b))
			for _, g := range got {
				if math.Float64bits(g) != math.Float64bits(want) {
					t.Fatalf("%s(%v (%#x), %v (%#x)) = %v (%#x), want %v (%#x)", op.name,
						c.a, math.Float64bits(c.a), c.b, math.Float64bits(c.b), g, math.Float64bits(g), want, math.Float64bits(want))
				}
			}
		}
	}
}

// The oracle against the Go builtin, ndarray's own expression: identical bits
// whenever the result is a number, a NaN wherever the builtin gives one. The NaN
// payload the builtin picks is the target's (an arm64 FMAX quiets a signalling
// NaN and orders two NaNs its own way), so the two are only counted.
func TestPairwiseBuiltin(t *testing.T) {
	edges := maxminEdges()
	rng := rand.New(rand.NewSource(83))
	for i := 0; i < 1<<16; i++ {
		edges = append(edges, math.Float64frombits(rng.Uint64()))
	}
	diff := map[string]int{}
	for _, op := range maxminOps {
		for i := 0; i < 1<<16; i++ {
			x, y := edges[rng.Intn(len(edges))], edges[rng.Intn(len(edges))]
			if i < len(maxminEdges())*len(maxminEdges()) {
				x, y = maxminEdges()[i%len(maxminEdges())], maxminEdges()[i/len(maxminEdges())]
			}
			w, b := op.want(x, y), op.bi(x, y)
			if math.IsNaN(w) != math.IsNaN(b) {
				t.Fatalf("%s(%v, %v): oracle %v, builtin %v", op.name, x, y, w, b)
			}
			if math.Float64bits(w) != math.Float64bits(b) {
				if !math.IsNaN(w) {
					t.Fatalf("%s(%#x, %#x): oracle %#x, builtin %#x", op.name, math.Float64bits(x), math.Float64bits(y), math.Float64bits(w), math.Float64bits(b))
				}
				k := op.name + " one NaN operand"
				if math.IsNaN(x) && math.IsNaN(y) {
					k = op.name + " two NaN operands"
				}
				diff[k]++
			}
		}
	}
	t.Logf("NaN results where the builtin's payload differs from the portable rule: %v", diff)
}

// FuzzPairwise: any bytes read as pairs of float64 words, either operation, any
// alignment: the kernel agrees with the rule, out of place and in place.
func FuzzPairwise(f *testing.F) {
	f.Add(uint8(0), uint8(0), []byte{})
	f.Add(uint8(1), uint8(1), []byte("zeros and NaNs: 0, -0, nan, +Inf, tail....."))
	f.Fuzz(func(t *testing.T, which, off uint8, data []byte) {
		op := maxminOps[int(which)%len(maxminOps)]
		n := len(data) / 16
		if n > 96 {
			n = 96
		}
		a, _ := fillE(n, int(off%4), rand.New(rand.NewSource(0)), false)
		b, _ := fillE(n, int(off%4+1)%4, rand.New(rand.NewSource(0)), false)
		word := func(p int) float64 {
			var u uint64
			for j := 0; j < 8; j++ {
				u |= uint64(data[p+j]) << (8 * j)
			}
			return math.Float64frombits(u)
		}
		for i := range a {
			a[i], b[i] = word(16*i), word(16*i+8)
		}
		dst := make([]float64, n)
		op.f(dst, a, b)
		checkMaxMin(t, op, "fuzz", dst, a, b, n, 0)
		ca := append([]float64(nil), a...)
		op.f(ca, ca, b)
		checkMaxMin(t, op, "fuzz in place", ca, a, b, n, 0)
	})
}
