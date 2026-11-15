package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func cdotLengths() []int { // complex numbers
	var l []int
	for n := 0; n <= 70; n++ {
		l = append(l, n)
	}
	return append(l, 127, 128, 129, 255, 256, 257, 1000, 1024, 4097)
}

func cdotInputs(rng *rand.Rand, floats int) []float64 {
	c := make([]float64, floats)
	for i := range c {
		c[i] = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(7)-3))
	}
	return c
}

// cdotExact is Σ conj(a_k) b_k in 300-bit arithmetic, and the sums of the absolute values
// of the products of the real part and of the imaginary part.
func cdotExact(a, b []float64) (re, im, magRe, magIm float64) {
	sr, si := new(big.Float).SetPrec(300), new(big.Float).SetPrec(300)
	f := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	for i := 0; i+1 < len(a); i += 2 {
		ar, ai, br, bi := f(a[i]), f(a[i+1]), f(b[i]), f(b[i+1])
		sr.Add(sr, new(big.Float).SetPrec(300).Mul(ar, br))
		sr.Add(sr, new(big.Float).SetPrec(300).Mul(ai, bi))
		si.Add(si, new(big.Float).SetPrec(300).Mul(ar, bi))
		si.Sub(si, new(big.Float).SetPrec(300).Mul(ai, br))
		magRe += math.Abs(a[i]*b[i]) + math.Abs(a[i+1]*b[i+1])
		magIm += math.Abs(a[i]*b[i+1]) + math.Abs(a[i+1]*b[i])
	}
	re, _ = sr.Float64()
	im, _ = si.Float64()
	return
}

const cdotU = 0x1p-53

// cdotBound is the stated bound, in units of the sum of the absolute products. The
// portable path is one serial chain of 2n fused multiply-adds per part, so a product can
// meet 2n roundings: (2n+2)u. The vector paths run chains of about n/4 steps (one per block
// of four complex numbers, in every lane) and then fold the lanes (at most 4 additions) and
// add a serial tail of at most three complex numbers (6 steps and an addition): (n/4+8)u.
func cdotBound(n int, vector bool) float64 {
	if vector {
		return (float64(n)/4 + 8) * cdotU
	}
	return (2*float64(n) + 2) * cdotU
}

func cdotCheck(t *testing.T, name string, n int, gr, gi float64, a, b []float64, vector bool) {
	t.Helper()
	er, ei, mr, mi := cdotExact(a, b)
	bound := cdotBound(n, vector)
	if d := math.Abs(gr - er); d > bound*mr {
		t.Fatalf("%s n=%d: real part %v, exact %v, error %.3e over the bound %.3e", name, n, gr, er, d, bound*mr)
	}
	if d := math.Abs(gi - ei); d > bound*mi {
		t.Fatalf("%s n=%d: imaginary part %v, exact %v, error %.3e over the bound %.3e", name, n, gi, ei, d, bound*mi)
	}
}

// TestCDotCBound: the public function is within cdotBound of the sum of the absolute
// products of the 300-bit value (the vector bound where a kernel runs, the portable one
// otherwise), and the portable loop within the portable bound, at every length from 0 to 70
// and at the sizes of a transform, for random data of several magnitudes and for b = a.
func TestCDotCBound(t *testing.T) {
	rng := rand.New(rand.NewSource(301))
	for _, n := range cdotLengths() {
		a, b := cdotInputs(rng, 2*n), cdotInputs(rng, 2*n+6)
		gr, gi := CDotC(a, b)
		cdotCheck(t, "CDotC", n, gr, gi, a, b[:2*n], cdotVector())
		tr, ti := cdotcTail(a, b[:2*n], 0)
		cdotCheck(t, "cdotcTail", n, tr, ti, a, b[:2*n], false)
		gr, gi = CDotC(a, a)
		cdotCheck(t, "CDotC a,a", n, gr, gi, a, a, cdotVector())
	}
}

// cdotAdversarial returns a and b whose serial chain rounds the same way at every step: the
// first complex number makes the real partial sum 2^52 (where the spacing of floats is 1),
// every later one adds 0.5+2^-30 twice (two fused steps per complex number), which rounds up
// by almost half a spacing each time. The exact sum is 2^52 + 2(n-1)(0.5+2^-30).
func cdotAdversarial(n int) (a, b []float64) {
	a, b = make([]float64, 2*n), make([]float64, 2*n)
	a[0], b[0] = 0x1p52, 1
	for k := 1; k < n; k++ {
		a[2*k], a[2*k+1] = 1, 1
		b[2*k], b[2*k+1] = 0.5+0x1p-30, 0.5+0x1p-30
	}
	return a, b
}

// TestCDotCAdversarial: with rounding errors that all have the same sign, the portable
// serial chain reaches about 2n units of 2^-53 of the sum of the absolute products, which is
// beyond (n+4) (the first version of the bound, wrong) and within the stated (2n+2); the
// public function stays within its own bound. Also data with massive cancellation: large
// products of both signs that sum to almost nothing, against the 300-bit value.
func TestCDotCAdversarial(t *testing.T) {
	for _, n := range []int{8, 16, 64, 256, 1000, 4097} {
		a, b := cdotAdversarial(n)
		er, _, mr, _ := cdotExact(a, b)
		tr, ti := cdotcTail(a, b, 0)
		got := math.Abs(tr-er) / (mr * cdotU)
		t.Logf("n=%d: the portable chain's error is %.2f units; the old bound n+4 = %d, the stated 2n+2 = %d", n, got, n+4, 2*n+2)
		if got <= float64(n+4) {
			t.Errorf("n=%d: the adversarial input reaches only %.2f units, not beyond n+4: it is not adversarial", n, got)
		}
		cdotCheck(t, "adversarial portable", n, tr, ti, a, b, false)
		gr, gi := CDotC(a, b)
		cdotCheck(t, "adversarial public", n, gr, gi, a, b, cdotVector())
	}
	rng := rand.New(rand.NewSource(305))
	for _, n := range []int{8, 33, 256, 1000, 4097} {
		a, b := make([]float64, 2*n), make([]float64, 2*n)
		for k := 0; k < n; k++ {
			x, y := rng.NormFloat64()*1e8, rng.NormFloat64()*1e8
			a[2*k], a[2*k+1] = x, y
			b[2*k], b[2*k+1] = 1+rng.Float64(), 1+rng.Float64()
			if k%2 == 1 { // the pair of the previous number with the signs of the products reversed
				a[2*k], a[2*k+1] = -a[2*k-2]*b[2*k-2]/b[2*k], -a[2*k-1]*b[2*k-1]/b[2*k+1]
			}
		}
		gr, gi := CDotC(a, b)
		cdotCheck(t, "cancellation public", n, gr, gi, a, b, cdotVector())
		tr, ti := cdotcTail(a, b, 0)
		cdotCheck(t, "cancellation portable", n, tr, ti, a, b, false)
	}
}

func TestCDotCPanicsAndBounds(t *testing.T) {
	mustPanic := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			r, _ := recover().(string)
			if r != want {
				t.Errorf("%s: panic %q, want %q", name, r, want)
			}
		}()
		f()
	}
	mustPanic("odd", "floats: CDotC: odd len(a)", func() { CDotC(make([]float64, 3), make([]float64, 4)) })
	mustPanic("short b", "floats: CDotC: len(b) < len(a)", func() { CDotC(make([]float64, 4), make([]float64, 3)) })
	if re, im := CDotC(nil, nil); re != 0 || im != 0 {
		t.Errorf("empty: %v %v", re, im)
	}
	// b past len(a) is not read, the inputs are not changed
	rng := rand.New(rand.NewSource(302))
	for _, n := range cdotLengths() {
		a, b := cdotInputs(rng, 2*n), cdotInputs(rng, 2*n)
		a0, b0 := append([]float64(nil), a...), append([]float64(nil), b...)
		long := append(append([]float64(nil), b...), math.NaN(), math.NaN(), math.NaN(), math.NaN())
		r1, i1 := CDotC(a, b)
		r2, i2 := CDotC(a, long)
		if r1 != r2 || i1 != i2 {
			t.Fatalf("n=%d: the result depends on b past len(a)", n)
		}
		for i := range a {
			if a[i] != a0[i] || b[i] != b0[i] {
				t.Fatalf("n=%d: an input changed", n)
			}
		}
	}
}

// FuzzCDotC: for any length 0 to 130 complex numbers and any finite bytes, the public
// function is within the bound of the 300-bit value.
func FuzzCDotC(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3})
	f.Add(uint8(3), []byte("complex dot"))
	f.Add(uint8(40), []byte{255, 0, 128})
	f.Add(uint8(129), []byte{})
	f.Fuzz(func(t *testing.T, k uint8, data []byte) {
		n := int(k) % 131
		a, b := make([]float64, 2*n), make([]float64, 2*n)
		for i := range a {
			if len(data) > 0 {
				a[i] = float64(int8(data[i%len(data)])) * 0.37
				b[i] = float64(int8(data[(i*5+2)%len(data)])) * 1.1
			}
		}
		gr, gi := CDotC(a, b)
		cdotCheck(t, "CDotC", n, gr, gi, a, b, cdotVector())
	})
}

// BenchmarkCDotC: 38 complex numbers (the lag-1 correlation of an SBR predictor), 1,024
// and 4,096.
func BenchmarkCDotC(b *testing.B) {
	rng := rand.New(rand.NewSource(303))
	for _, n := range []int{38, 1024, 4096} {
		x, y := cdotInputs(rng, 2*n), cdotInputs(rng, 2*n)
		name := map[int]string{38: "38", 1024: "1024", 4096: "4096"}[n]
		b.Run("kernel/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cdotSinkRe, cdotSinkIm = CDotC(x, y)
			}
		})
		b.Run("portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cdotSinkRe, cdotSinkIm = cdotcTail(x, y, 0)
			}
		})
	}
}

var cdotSinkRe, cdotSinkIm float64
