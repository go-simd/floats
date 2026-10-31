package floats

import (
	"math"
	"math/rand"
	"testing"
)

// Signed zeros are made with Copysign: the literal -0.0 is folded to +0 by Go.
var negZero = math.Copysign(0, -1)

// Negative and signaling NaNs: sign bit set / quiet bit clear.
var (
	negNaN  = math.Float64frombits(0xfff8000000000123)
	sigNaN  = math.Float64frombits(0x7ff0000000000001)
	nsigNaN = math.Float64frombits(0xfff0000000000001)
)

var nanKinds = []float64{math.NaN(), negNaN, sigNaN, nsigNaN}

// elemSpecials extends specials with the NaN kinds above.
var elemSpecials = append(append([]float64(nil), specials...), negNaN, sigNaN, nsigNaN)

// fillE is fill with elemSpecials.
func fillE(n, off int, rng *rand.Rand, withSpecials bool) (s, buf []float64) {
	s, buf = fill(n, off, rng, false)
	if withSpecials {
		for i := range s {
			if rng.Intn(6) == 0 {
				s[i] = elemSpecials[rng.Intn(len(elemSpecials))]
			}
		}
	}
	return s, buf
}

func exactBits(t *testing.T, what string, got, want []float64, n, off int) {
	t.Helper()
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s n=%d off=%d: [%d] = %v (%#x), want %v (%#x)", what, n, off, i,
				got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

func TestMulBitIdentical(t *testing.T) {
	forEachCase(t, 21, func(n, off int, sp bool, rng *rand.Rand) {
		a, _ := fillE(n, off, rng, sp)
		b, _ := fillE(n, (off+1)%4, rng, sp)
		dst, buf := fillE(n, (off+2)%4, rng, sp)
		want := make([]float64, n)
		for i := range want {
			want[i] = a[i] * b[i]
		}
		Mul(dst, a, b)
		checkSame(t, "Mul", dst, want, n, off)
		checkGuard(t, "Mul", buf, n, (off+2)%4)
		mulRef(dst, a, b)
		checkSame(t, "mulRef", dst, want, n, off)
	})
}

func TestAxpby3BitIdentical(t *testing.T) {
	forEachCase(t, 22, func(n, off int, sp bool, rng *rand.Rand) {
		for _, al := range alphas(rng) {
			be, ga := rng.NormFloat64(), alphas(rng)[rng.Intn(7)]
			a, _ := fillE(n, off, rng, sp)
			b, _ := fillE(n, (off+1)%4, rng, sp)
			c, _ := fillE(n, (off+3)%4, rng, sp)
			dst, buf := fillE(n, (off+2)%4, rng, sp)
			want := make([]float64, n)
			for i := range want { // the documented definition, spelled out
				want[i] = math.FMA(ga, c[i], math.FMA(be, b[i], al*a[i]))
			}
			Axpby3(dst, al, a, be, b, ga, c)
			checkSame(t, "Axpby3", dst, want, n, off)
			checkGuard(t, "Axpby3", buf, n, (off+2)%4)
		}
	})
}

// TestAxpby3IsFused shows the combination rounds like the math.FMA chain and
// not like the unfused expression: with t = −1, β = 1+2⁻³⁰ and b = 1−2⁻³⁰,
// β·b = 1−2⁻⁶⁰ rounds to 1 on its own, so the unfused sum is 0 and the fused
// one keeps −2⁻⁶⁰. And against the unfused form on random data it stays within
// a few ULP of Σ|term|.
func TestAxpby3IsFused(t *testing.T) {
	be := 1 + math.Exp2(-30)
	dst := make([]float64, 9)
	one := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1}
	neg := []float64{-1, -1, -1, -1, -1, -1, -1, -1, -1}
	b := []float64{}
	for range dst {
		b = append(b, 1-math.Exp2(-30))
	}
	Axpby3(dst, 1, neg, be, b, 0, one)
	for i, v := range dst {
		if v != -math.Exp2(-60) {
			t.Fatalf("Axpby3 fused [%d] = %v, want %v", i, v, -math.Exp2(-60))
		}
	}

	rng := rand.New(rand.NewSource(23))
	const eps = 0x1p-53
	for _, n := range sizes {
		a, b, c := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		al, be, ga := rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()
		dst := make([]float64, n)
		Axpby3(dst, al, a, be, b, ga, c)
		for i := range dst {
			unfused := al*a[i] + be*b[i] + ga*c[i]
			scale := math.Abs(al*a[i]) + math.Abs(be*b[i]) + math.Abs(ga*c[i])
			if math.Abs(dst[i]-unfused) > 4*eps*scale {
				t.Fatalf("Axpby3 n=%d [%d] = %v, unfused %v, |terms| %v", n, i, dst[i], unfused, scale)
			}
		}
	}
}

func TestClampBitIdentical(t *testing.T) {
	forEachCase(t, 24, func(n, off int, sp bool, rng *rand.Rand) {
		v, _ := fillE(n, off, rng, sp)
		lo, _ := fillE(n, (off+1)%4, rng, sp)
		hi, _ := fillE(n, (off+3)%4, rng, sp)
		dst, buf := fillE(n, (off+2)%4, rng, sp)
		want := make([]float64, n)
		for i := range want { // the documented definition, spelled out
			r := v[i]
			if lo[i] > r {
				r = lo[i]
			}
			if hi[i] < r {
				r = hi[i]
			}
			want[i] = r
		}
		Clamp(dst, v, lo, hi)
		exactBits(t, "Clamp", dst, want, n, off) // NaN bits included
		checkGuard(t, "Clamp", buf, n, (off+2)%4)
		clampRef(dst, v, lo, hi)
		exactBits(t, "clampRef", dst, want, n, off)
	})
}

// TestClampSemantics pins the NaN, signed-zero and crossed-bounds rules one
// case at a time, over every position of a 0..67 element vector so each loop
// stage (x16, x4, scalar) sees each case.
func TestClampSemantics(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name         string
		v, lo, hi    float64
		want         float64
		wantNaNBits  bool // result must carry v's exact bits
		wantSignBits bool // compare with Float64bits (signed zero)
	}{
		{"inside", 0.5, 0, 1, 0.5, false, false},
		{"below", -3, 0, 1, 0, false, false},
		{"above", 3, 0, 1, 1, false, false},
		{"on lo", 0, 0, 1, 0, false, false},
		{"crossed: hi wins", 0.5, 2, 1, 1, false, false},
		{"unbounded", 1e300, -inf, inf, 1e300, false, false},
		{"v=+Inf clamped", inf, 0, 1, 1, false, false},
		{"v=-Inf clamped", -inf, 0, 1, 0, false, false},
		{"NaN v", math.NaN(), 0, 1, math.NaN(), true, false},
		{"negative NaN v", negNaN, 0, 1, negNaN, true, false},
		{"signaling NaN v", sigNaN, 0, 1, sigNaN, true, false},
		{"NaN v, crossed", negNaN, 2, 1, negNaN, true, false},
		{"NaN lo ignored", -5, math.NaN(), 1, -5, false, false},
		{"negative NaN lo ignored", 5, negNaN, 1, 1, false, false},
		{"NaN hi ignored", 5, 0, math.NaN(), 5, false, false},
		{"negative NaN hi ignored", -5, 0, negNaN, 0, false, false},
		{"both bounds NaN", 7, math.NaN(), negNaN, 7, false, false},
		{"-0 vs lo=+0 keeps -0", negZero, 0, 1, negZero, false, true},
		{"+0 vs lo=-0 keeps +0", 0, negZero, 1, 0, false, true},
		{"-0 vs hi=+0 keeps -0", negZero, -1, 0, negZero, false, true},
		{"+0 vs hi=-0 keeps +0", 0, -1, negZero, 0, false, true},
		{"-0 inside", negZero, -1, 1, negZero, false, true},
		{"v<lo gives lo=-0", -1, negZero, 1, negZero, false, true},
		{"v>hi gives hi=-0", 1, -1, negZero, negZero, false, true},
	}
	for _, c := range cases {
		for n := 1; n <= 67; n++ {
			for pos := 0; pos < n; pos++ {
				v, lo, hi := make([]float64, n), make([]float64, n), make([]float64, n)
				for i := range v { // neighbours: an interior value, so a bad lane shows
					v[i], lo[i], hi[i] = 0.25, 0, 1
				}
				v[pos], lo[pos], hi[pos] = c.v, c.lo, c.hi
				dst := make([]float64, n)
				Clamp(dst, v, lo, hi)
				for i := range dst {
					want := 0.25
					if i == pos {
						want = c.want
					}
					got := dst[i]
					switch {
					case i == pos && (c.wantNaNBits || c.wantSignBits):
						if math.Float64bits(got) != math.Float64bits(want) {
							t.Fatalf("%s n=%d pos=%d: got %#x want %#x", c.name, n, pos,
								math.Float64bits(got), math.Float64bits(want))
						}
					case math.IsNaN(want):
						if !math.IsNaN(got) {
							t.Fatalf("%s n=%d pos=%d: got %v want NaN", c.name, n, pos, got)
						}
					case got != want:
						t.Fatalf("%s n=%d pos=%d i=%d: got %v want %v", c.name, n, pos, i, got, want)
					}
				}
			}
		}
	}
}

func naiveMaxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		a := math.Abs(x)
		if a != a {
			return math.NaN()
		}
		if a > m {
			m = a
		}
	}
	return m
}

func TestMaxAbsBitIdentical(t *testing.T) {
	forEachCase(t, 25, func(n, off int, sp bool, rng *rand.Rand) {
		v, _ := fillE(n, off, rng, sp)
		want := naiveMaxAbs(v)
		got := MaxAbs(v)
		if !sameBits(got, want) {
			t.Fatalf("MaxAbs n=%d off=%d: %v (%#x), want %v (%#x)", n, off, got,
				math.Float64bits(got), want, math.Float64bits(want))
		}
		if r := maxAbsRef(v); !sameBits(r, want) && !(math.IsNaN(r) && math.IsNaN(want)) {
			t.Fatalf("maxAbsRef n=%d: %v want %v", n, r, want)
		}
	})
}

// TestMaxAbsPositions plants one value at every position of every length 0..67
// (so each loop stage and each lane sees it) among zeros, negative zeros and
// small values: a planted maximum is found, any NaN of either sign wins over an
// infinity, and an all-zero slice gives +0, never -0.
func TestMaxAbsPositions(t *testing.T) {
	if got := MaxAbs(nil); got != 0 || math.Signbit(got) {
		t.Fatalf("MaxAbs(nil) = %v", got)
	}
	for n := 1; n <= 67; n++ {
		zeros := make([]float64, n)
		for i := range zeros {
			zeros[i] = negZero
		}
		if got := MaxAbs(zeros); got != 0 || math.Signbit(got) {
			t.Fatalf("MaxAbs(-0 × %d) = %v (signbit %v)", n, got, math.Signbit(got))
		}
		for pos := 0; pos < n; pos++ {
			plant := func(x float64, others float64) []float64 {
				v := make([]float64, n)
				for i := range v {
					v[i] = others * float64(i%3-1)
				}
				v[pos] = x
				return v
			}
			for _, x := range []float64{7.5, -7.5, math.Inf(1), math.Inf(-1), math.MaxFloat64, -math.MaxFloat64} {
				want := math.Abs(x)
				if got := MaxAbs(plant(x, 1)); got != want {
					t.Fatalf("MaxAbs n=%d pos=%d planted %v: %v", n, pos, x, got)
				}
			}
			for _, nan := range nanKinds {
				for _, others := range []float64{1, 1e300} {
					v := plant(nan, others)
					if got := MaxAbs(v); !math.IsNaN(got) {
						t.Fatalf("MaxAbs n=%d pos=%d NaN %#x: %v", n, pos, math.Float64bits(nan), got)
					}
					// NaN must beat an infinity placed at another position.
					if n > 1 {
						v[(pos+1)%n] = math.Inf(1)
						if got := MaxAbs(v); !math.IsNaN(got) {
							t.Fatalf("MaxAbs n=%d pos=%d NaN+Inf: %v", n, pos, got)
						}
						v[(pos+1)%n] = math.Inf(-1)
						if got := MaxAbs(v); !math.IsNaN(got) {
							t.Fatalf("MaxAbs n=%d pos=%d NaN-Inf: %v", n, pos, got)
						}
					}
				}
			}
		}
	}
	if got := MaxAbs([]float64{negNaN}); math.Float64bits(got) != math.Float64bits(math.NaN()) {
		t.Fatalf("MaxAbs returns %#x, want the canonical math.NaN()", math.Float64bits(got))
	}
}

// TestElemAliasing covers dst being the same slice as an input.
func TestElemAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(26))
	for n := 0; n <= 67; n++ {
		a, b, c := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		lo, hi := randF64(n, rng), randF64(n, rng)
		clone := func(s []float64) []float64 { return append([]float64(nil), s...) }

		want := make([]float64, n)
		mulRef(want, a, b)
		x := clone(a)
		Mul(x, x, b)
		checkSame(t, "Mul(a,a,b)", x, want, n, 0)
		y := clone(b)
		Mul(y, a, y)
		checkSame(t, "Mul(b,a,b)", y, want, n, 0)

		axpby3Ref(want, 0.5, a, -1.5, b, 2.5, c)
		x, z := clone(a), clone(c)
		Axpby3(x, 0.5, x, -1.5, b, 2.5, c)
		checkSame(t, "Axpby3 dst=a", x, want, n, 0)
		Axpby3(z, 0.5, a, -1.5, b, 2.5, z)
		checkSame(t, "Axpby3 dst=c", z, want, n, 0)

		clampRef(want, a, lo, hi)
		x, h := clone(a), clone(hi)
		Clamp(x, x, lo, hi)
		exactBits(t, "Clamp dst=v", x, want, n, 0)
		Clamp(h, a, lo, h)
		exactBits(t, "Clamp dst=hi", h, want, n, 0)
	}
}

// TestElemLongerSlices checks that only the first len(a) (len(v)) elements take
// part and nothing past them is written.
func TestElemLongerSlices(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6, 7}
	c := []float64{1, 1, 1, 1, 1}
	lo := []float64{2, 2, 2, 2}
	hi := []float64{2.5, 2.5, 2.5, 9}
	dst := []float64{-1, -1, -1, -1, -1, -1}
	keep := func(want []float64) {
		t.Helper()
		if !equalF64(dst, want) {
			t.Fatalf("got %v want %v", dst, want)
		}
	}
	Mul(dst, a, b)
	keep([]float64{4, 10, 18, -1, -1, -1})
	Axpby3(dst, 2, a, 1, b, -1, c)
	keep([]float64{5, 8, 11, -1, -1, -1})
	Clamp(dst, a, lo, hi)
	keep([]float64{2, 2, 2.5, -1, -1, -1})
}

func TestElemPanics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		f()
	}
	two, one := []float64{1, 2}, []float64{1}
	mustPanic("Mul b", func() { Mul(two, two, one) })
	mustPanic("Mul dst", func() { Mul(one, two, two) })
	mustPanic("Axpby3 b", func() { Axpby3(two, 1, two, 1, one, 1, two) })
	mustPanic("Axpby3 c", func() { Axpby3(two, 1, two, 1, two, 1, one) })
	mustPanic("Axpby3 dst", func() { Axpby3(one, 1, two, 1, two, 1, two) })
	mustPanic("Clamp lo", func() { Clamp(two, two, one, two) })
	mustPanic("Clamp hi", func() { Clamp(two, two, two, one) })
	mustPanic("Clamp dst", func() { Clamp(one, two, two, two) })
	// A zero-length call is fine whatever the other lengths.
	Mul(nil, nil, nil)
	Axpby3(nil, 1, nil, 1, nil, 1, nil)
	Clamp(nil, nil, nil, nil)
}
