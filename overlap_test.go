package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

const overlapPrec = 200 // bits of the oracle

func bigOf(x float64) *big.Float { return new(big.Float).SetPrec(overlapPrec).SetFloat64(x) }

// overlapOracle returns, for step i, the exact values of the two outputs and the
// sums of the magnitudes of their two products.
func overlapOracle(a, b, win []float64, i int) (lo, hi *big.Float, sLo, sHi float64) {
	n := len(a)
	j := n - 1 - i
	s0, s1 := bigOf(a[i]), bigOf(b[j])
	wi, wj := bigOf(win[i]), bigOf(win[2*n-1-i])
	p := func(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(overlapPrec).Mul(x, y) }
	lo = new(big.Float).SetPrec(overlapPrec).Sub(p(s0, wj), p(s1, wi))
	hi = new(big.Float).SetPrec(overlapPrec).Add(p(s0, wi), p(s1, wj))
	sLo = math.Abs(a[i]*win[2*n-1-i]) + math.Abs(b[j]*win[i])
	sHi = math.Abs(a[i]*win[i]) + math.Abs(b[j]*win[2*n-1-i])
	return
}

func overlapInputs(rng *rand.Rand, n int) (a, b, win []float64) {
	a, b, win = make([]float64, n), make([]float64, n), make([]float64, 2*n)
	for i := range a {
		a[i], b[i] = rng.NormFloat64()*100, rng.NormFloat64()*100
	}
	for i := range win { // a sine window: the shape the codecs use
		win[i] = math.Sin(math.Pi / float64(2*n) * (float64(i) + 0.5))
	}
	return
}

var overlapSizes = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 127, 128, 129, 512, 1000, 1024}

// checkBound requires every output within 2 ulp-of-the-products of the exact
// value: |got - exact| <= 2 * 2^-prec * (|p1| + |p2|).
func checkBound(t *testing.T, name string, n int, got []float64, a, b, win []float64, prec int) {
	t.Helper()
	u := math.Ldexp(1, -prec)
	for i := 0; i < n; i++ {
		lo, hi, sLo, sHi := overlapOracle(a, b, win, i)
		for _, c := range []struct {
			idx   int
			exact *big.Float
			sum   float64
		}{{i, lo, sLo}, {2*n - 1 - i, hi, sHi}} {
			d := new(big.Float).SetPrec(overlapPrec).Sub(bigOf(got[c.idx]), c.exact)
			e, _ := d.Float64()
			if math.Abs(e) > 2*u*c.sum {
				t.Fatalf("%s n=%d: output %d is %v, exact %v: error %g exceeds %g", name, n, c.idx, got[c.idx], c.exact, e, 2*u*c.sum)
			}
		}
	}
}

func TestOverlapWindowBound(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, n := range overlapSizes {
		a, b, win := overlapInputs(rng, n)
		dst := make([]float64, 2*n)
		OverlapWindow(dst, a, b, win)
		checkBound(t, "OverlapWindow", n, dst, a, b, win, 53)
		ref := make([]float64, 2*n)
		overlapRef64(ref, a, b, win)
		checkBound(t, "overlapRef64", n, ref, a, b, win, 53)

		// float32: the oracle sees the float32 values
		a32, b32, w32 := to32(a), to32(b), to32(win)
		d32 := make([]float32, 2*n)
		Float32OverlapWindow(d32, a32, b32, w32)
		checkBound(t, "Float32OverlapWindow", n, to64(d32), to64(a32), to64(b32), to64(w32), 24)
		r32 := make([]float32, 2*n)
		overlapRef32(r32, a32, b32, w32)
		checkBound(t, "overlapRef32", n, to64(r32), to64(a32), to64(b32), to64(w32), 24)
	}
}

func to32(x []float64) []float32 {
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = float32(v)
	}
	return y
}

func to64(x []float32) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = float64(v)
	}
	return y
}

func TestOverlapWindowPanics(t *testing.T) {
	for _, bad := range []struct{ a, b, w, d int }{{4, 3, 8, 8}, {4, 4, 7, 8}, {4, 4, 8, 7}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("OverlapWindow%v: no panic", bad)
				}
			}()
			OverlapWindow(make([]float64, bad.d), make([]float64, bad.a), make([]float64, bad.b), make([]float64, bad.w))
		}()
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Float32OverlapWindow%v: no panic", bad)
				}
			}()
			Float32OverlapWindow(make([]float32, bad.d), make([]float32, bad.a), make([]float32, bad.b), make([]float32, bad.w))
		}()
	}
	OverlapWindow(nil, nil, nil, nil) // n = 0: nothing
	Float32OverlapWindow(nil, nil, nil, nil)
}

// TestOverlapWindowStaysInBounds: nothing is written outside dst[:2n], and the
// inputs are not modified, whatever the size (the tail of the kernels).
func TestOverlapWindowStaysInBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	for _, n := range overlapSizes {
		a, b, win := overlapInputs(rng, n)
		a0, b0, w0 := append([]float64(nil), a...), append([]float64(nil), b...), append([]float64(nil), win...)
		buf := make([]float64, 2*n+16)
		for i := range buf {
			buf[i] = 12345
		}
		OverlapWindow(buf[8:8+2*n], a, b, win)
		for i := range buf {
			if (i < 8 || i >= 8+2*n) && buf[i] != 12345 {
				t.Fatalf("n=%d: wrote outside dst at %d", n, i)
			}
		}
		for i := range a {
			if a[i] != a0[i] || b[i] != b0[i] {
				t.Fatalf("n=%d: an input changed", n)
			}
		}
		for i := range win {
			if win[i] != w0[i] {
				t.Fatalf("n=%d: the window changed", n)
			}
		}
		buf32 := make([]float32, 2*n+16)
		for i := range buf32 {
			buf32[i] = 12345
		}
		Float32OverlapWindow(buf32[8:8+2*n], to32(a), to32(b), to32(win))
		for i := range buf32 {
			if (i < 8 || i >= 8+2*n) && buf32[i] != 12345 {
				t.Fatalf("float32 n=%d: wrote outside dst at %d", n, i)
			}
		}
	}
}

func BenchmarkOverlapWindow(b *testing.B) {
	rng := rand.New(rand.NewSource(34))
	for _, n := range []int{64, 1024} {
		a, bb, win := overlapInputs(rng, n)
		dst := make([]float64, 2*n)
		a32, b32, w32, d32 := to32(a), to32(bb), to32(win), make([]float32, 2*n)
		name := map[int]string{64: "64", 1024: "1024"}[n]
		b.Run("float64/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				OverlapWindow(dst, a, bb, win)
			}
		})
		b.Run("float64-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				overlapRef64(dst, a, bb, win)
			}
		})
		b.Run("float32/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Float32OverlapWindow(d32, a32, b32, w32)
			}
		})
		b.Run("float32-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				overlapRef32(d32, a32, b32, w32)
			}
		})
	}
}
