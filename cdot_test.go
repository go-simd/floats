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

func cdotCheck(t *testing.T, name string, n int, gr, gi float64, a, b []float64) {
	t.Helper()
	er, ei, mr, mi := cdotExact(a, b)
	bound := float64(n+4) * 0x1p-53
	if d := math.Abs(gr - er); d > bound*mr {
		t.Fatalf("%s n=%d: real part %v, exact %v, error %.3e over the bound %.3e", name, n, gr, er, d, bound*mr)
	}
	if d := math.Abs(gi - ei); d > bound*mi {
		t.Fatalf("%s n=%d: imaginary part %v, exact %v, error %.3e over the bound %.3e", name, n, gi, ei, d, bound*mi)
	}
}

// TestCDotCBound: the kernel, the public function and the portable loop are within
// (n+4)·2^-53 of the sum of the absolute products of the 300-bit value, at every length
// from 0 to 70 and at the sizes of a transform, for random data of several magnitudes and
// for data with massive cancellation.
func TestCDotCBound(t *testing.T) {
	rng := rand.New(rand.NewSource(301))
	for _, n := range cdotLengths() {
		a, b := cdotInputs(rng, 2*n), cdotInputs(rng, 2*n+6)
		gr, gi := CDotC(a, b)
		cdotCheck(t, "CDotC", n, gr, gi, a, b[:2*n])
		tr, ti := cdotcTail(a, b[:2*n], 0)
		cdotCheck(t, "cdotcTail", n, tr, ti, a, b[:2*n])
		// b = a: the real part is the sum of the squared magnitudes, the imaginary part
		// cancels exactly in the definition
		gr, gi = CDotC(a, a)
		cdotCheck(t, "CDotC a,a", n, gr, gi, a, a)
		if gi != 0 && n > 0 {
			// not exactly zero in general (reordered fused sums): bounded by the check above
			_ = gi
		}
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
		cdotCheck(t, "CDotC", n, gr, gi, a, b)
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
