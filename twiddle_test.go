package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

var twiddleSizes = []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 30, 32, 34, 62, 64, 66, 126, 128, 130, 254, 256, 510, 512, 1000, 1024, 2048}

// twiddleInputs: x (real samples), c (complex values) and a table of unit
// complex numbers, the shape of a DCT-IV's twiddles.
func twiddleInputs(rng *rand.Rand, n int) (x, c, tw []float64) {
	x, c, tw = make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range x {
		x[i], c[i] = rng.NormFloat64()*100, rng.NormFloat64()*100
	}
	for m := 0; m < n/2; m++ {
		s, co := math.Sincos(-math.Pi * float64(m) / float64(n))
		tw[2*m], tw[2*m+1] = co, s
	}
	return
}

func bmul(x, y float64) *big.Float {
	return new(big.Float).SetPrec(overlapPrec).Mul(bigOf(x), bigOf(y))
}

// gatherOracle: the exact re and im of output pair m, and the sums of the
// magnitudes of their two products.
func gatherOracle(x, tw []float64, m int) (re, im *big.Float, sRe, sIm float64) {
	n := len(tw)
	e, o := x[2*m], x[n-1-2*m]
	wr, wi := tw[2*m], tw[2*m+1]
	re = new(big.Float).SetPrec(overlapPrec).Sub(bmul(e, wr), bmul(o, wi))
	im = new(big.Float).SetPrec(overlapPrec).Add(bmul(e, wi), bmul(o, wr))
	return re, im, math.Abs(e*wr) + math.Abs(o*wi), math.Abs(e*wi) + math.Abs(o*wr)
}

// scatterOracle: the exact value of dst[2m] and dst[n-1-2m] (with f) and the sums.
func scatterOracle(c, tw []float64, f float64, m int) (re, im *big.Float, sRe, sIm float64) {
	cr, ci := c[2*m], c[2*m+1]
	wr, wi := tw[2*m], tw[2*m+1]
	zr := new(big.Float).SetPrec(overlapPrec).Sub(bmul(wr, cr), bmul(wi, ci))
	zi := new(big.Float).SetPrec(overlapPrec).Add(bmul(wr, ci), bmul(wi, cr))
	fb := bigOf(f)
	re = zr.Mul(zr, fb)
	im = zi.Mul(zi, fb)
	im.Neg(im)
	return re, im, math.Abs(f) * (math.Abs(wr*cr) + math.Abs(wi*ci)), math.Abs(f) * (math.Abs(wr*ci) + math.Abs(wi*cr))
}

func errOf(got float64, exact *big.Float) float64 {
	d := new(big.Float).SetPrec(overlapPrec).Sub(bigOf(got), exact)
	e, _ := d.Float64()
	return math.Abs(e)
}

func exactAbs(x *big.Float) float64 {
	v, _ := x.Float64()
	return math.Abs(v)
}

const u53 = 0x1p-53

func TestCMulGatherBound(t *testing.T) { gatherBoundSweep(t) }

func gatherBoundSweep(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	for _, n := range twiddleSizes {
		x, _, tw := twiddleInputs(rng, n)
		x32 := to32(x)
		for name, got := range map[string][]float64{
			"CMulGather":       func() []float64 { d := make([]float64, n); CMulGather(d, x, tw); return d }(),
			"cmulGatherTail":   func() []float64 { d := make([]float64, n); cmulGatherTail(d, x, tw, 0); return d }(),
			"CMulGatherFrom32": func() []float64 { d := make([]float64, n); CMulGatherFrom32(d, x32, tw); return d }(),
		} {
			xs := x
			if name == "CMulGatherFrom32" {
				xs = to64(x32) // the oracle sees the float32 values
			}
			for m := 0; m < n/2; m++ {
				re, im, sRe, sIm := gatherOracle(xs, tw, m)
				if e := errOf(got[2*m], re); e > 4*u53*sRe {
					t.Fatalf("%s n=%d: re[%d] error %g exceeds %g", name, n, m, e, 4*u53*sRe)
				}
				if e := errOf(got[2*m+1], im); e > 4*u53*sIm {
					t.Fatalf("%s n=%d: im[%d] error %g exceeds %g", name, n, m, e, 4*u53*sIm)
				}
			}
		}
	}
}

func TestCMulScatterBound(t *testing.T) { scatterBoundSweep(t) }

func scatterBoundSweep(t *testing.T) {
	rng := rand.New(rand.NewSource(62))
	for _, n := range twiddleSizes {
		_, c, tw := twiddleInputs(rng, n)
		for _, f := range []float64{1, 0.37, -2.5, 0.5} {
			d := make([]float64, n)
			CMulScatter(d, c, tw, f)
			r := make([]float64, n)
			cmulScatterRef(r, c, tw, f)
			d32 := make([]float32, n)
			CMulScatterTo32(d32, c, tw, f)
			r32 := make([]float32, n)
			cmulScatter32Ref(r32, c, tw, f)
			for m := 0; m < n/2; m++ {
				re, im, sRe, sIm := scatterOracle(c, tw, f, m)
				for _, ch := range []struct {
					name  string
					got   float64
					exact *big.Float
					sum   float64
					idx   int
					f32   bool
				}{
					{"CMulScatter", d[2*m], re, sRe, 2 * m, false}, {"CMulScatter", d[n-1-2*m], im, sIm, n - 1 - 2*m, false},
					{"scatterRef", r[2*m], re, sRe, 2 * m, false}, {"scatterRef", r[n-1-2*m], im, sIm, n - 1 - 2*m, false},
					{"CMulScatterTo32", float64(d32[2*m]), re, sRe, 2 * m, true}, {"CMulScatterTo32", float64(d32[n-1-2*m]), im, sIm, n - 1 - 2*m, true},
					{"scatter32Ref", float64(r32[2*m]), re, sRe, 2 * m, true}, {"scatter32Ref", float64(r32[n-1-2*m]), im, sIm, n - 1 - 2*m, true},
				} {
					bound := 4 * u53 * ch.sum
					if ch.f32 {
						bound = 0x1p-24*exactAbs(ch.exact) + 8*u53*ch.sum
					}
					if e := errOf(ch.got, ch.exact); e > bound {
						t.Fatalf("%s n=%d f=%g: output %d error %g exceeds %g", ch.name, n, f, ch.idx, e, bound)
					}
				}
			}
		}
	}
}

func TestCMulPanicsAndBounds(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic", name)
			}
		}()
		f()
	}
	f8, f7, f32x8, f32x7 := make([]float64, 8), make([]float64, 7), make([]float32, 8), make([]float32, 7)
	mustPanic("gather odd tw", func() { CMulGather(f8, f8, f7) })
	mustPanic("gather short x", func() { CMulGather(f8, f7, f8) })
	mustPanic("gather short dst", func() { CMulGather(f7, f8, f8) })
	mustPanic("gather32 short x", func() { CMulGatherFrom32(f8, f32x7, f8) })
	mustPanic("gather32 short dst", func() { CMulGatherFrom32(f7, f32x8, f8) })
	mustPanic("gather32 odd tw", func() { CMulGatherFrom32(f8, f32x8, f7) })
	mustPanic("scatter odd tw", func() { CMulScatter(f8, f8, f7, 1) })
	mustPanic("scatter short c", func() { CMulScatter(f8, f7, f8, 1) })
	mustPanic("scatter short dst", func() { CMulScatter(f7, f8, f8, 1) })
	mustPanic("scatter32 short c", func() { CMulScatterTo32(f32x8, f7, f8, 1) })
	mustPanic("scatter32 short dst", func() { CMulScatterTo32(f32x7, f8, f8, 1) })
	mustPanic("scatter32 odd tw", func() { CMulScatterTo32(f32x8, f8, f7, 1) })
	CMulGather(nil, nil, nil) // n = 0: nothing
	CMulGatherFrom32(nil, nil, nil)
	CMulScatter(nil, nil, nil, 1)
	CMulScatterTo32(nil, nil, nil, 1)

	// Nothing is written outside dst[:n], the inputs are not modified, longer
	// slices are read only up to n.
	rng := rand.New(rand.NewSource(63))
	for _, n := range twiddleSizes {
		x, c, tw := twiddleInputs(rng, n)
		x0, c0, tw0 := append([]float64(nil), x...), append([]float64(nil), c...), append([]float64(nil), tw...)
		buf := make([]float64, n+16)
		buf32 := make([]float32, n+16)
		for i := range buf {
			buf[i], buf32[i] = 12345, 12345
		}
		long := append(append([]float64(nil), x...), 1e300, 1e300, 1e300, 1e300) // must not be read
		CMulGather(buf[8:8+n], long, tw)
		for i := range buf {
			if (i < 8 || i >= 8+n) && buf[i] != 12345 {
				t.Fatalf("CMulGather n=%d: wrote outside dst at %d", n, i)
			}
			buf[i] = 12345
		}
		CMulScatter(buf[8:8+n], c, tw, 0.5)
		CMulScatterTo32(buf32[8:8+n], c, tw, 0.5)
		for i := range buf {
			if (i < 8 || i >= 8+n) && (buf[i] != 12345 || buf32[i] != 12345) {
				t.Fatalf("scatter n=%d: wrote outside dst at %d", n, i)
			}
		}
		for i := range x {
			if x[i] != x0[i] || c[i] != c0[i] || tw[i] != tw0[i] {
				t.Fatalf("n=%d: an input changed", n)
			}
		}
		for i := 0; i < n; i++ {
			if math.IsNaN(buf[8+i]) || math.IsInf(buf[8+i], 0) || math.Abs(buf[8+i]) > 1e6 {
				t.Fatalf("n=%d: output %d is %g: an operand past n was read", n, i, buf[8+i])
			}
		}
	}
}

// FuzzCMulTwiddle: for any even length up to 130 and any bytes, the kernels and
// the portable loops agree to the sum of the magnitudes of the products, and
// nothing panics.
func FuzzCMulTwiddle(f *testing.F) {
	f.Add(uint8(2), []byte{1, 2, 3})
	f.Add(uint8(3), []byte("fuzz the twiddle kernels"))
	f.Add(uint8(33), []byte{255, 0, 128})
	f.Add(uint8(64), []byte{})
	f.Fuzz(func(t *testing.T, k uint8, data []byte) {
		n := 2 * (int(k)%65 + 1)
		at := func(i, mul int) float64 {
			if len(data) == 0 {
				return 0
			}
			return float64(int(data[(i*mul+1)%len(data)])-128) / 16
		}
		x, c, tw := make([]float64, n), make([]float64, n), make([]float64, n)
		for i := range x {
			x[i], c[i], tw[i] = at(i, 3), at(i, 5), at(i, 7)
		}
		got, ref := make([]float64, n), make([]float64, n)
		CMulGather(got, x, tw)
		cmulGatherTail(ref, x, tw, 0)
		for m := 0; m < n/2; m++ {
			sRe := math.Abs(x[2*m]*tw[2*m]) + math.Abs(x[n-1-2*m]*tw[2*m+1])
			sIm := math.Abs(x[2*m]*tw[2*m+1]) + math.Abs(x[n-1-2*m]*tw[2*m])
			if math.Abs(got[2*m]-ref[2*m]) > 8*u53*sRe || math.Abs(got[2*m+1]-ref[2*m+1]) > 8*u53*sIm {
				t.Fatalf("gather n=%d m=%d: %g %g, portable %g %g", n, m, got[2*m], got[2*m+1], ref[2*m], ref[2*m+1])
			}
		}
		CMulScatter(got, c, tw, 0.75)
		cmulScatterRef(ref, c, tw, 0.75)
		for m := 0; m < n/2; m++ {
			sRe := 0.75 * (math.Abs(tw[2*m]*c[2*m]) + math.Abs(tw[2*m+1]*c[2*m+1]))
			sIm := 0.75 * (math.Abs(tw[2*m]*c[2*m+1]) + math.Abs(tw[2*m+1]*c[2*m]))
			if math.Abs(got[2*m]-ref[2*m]) > 8*u53*sRe || math.Abs(got[n-1-2*m]-ref[n-1-2*m]) > 8*u53*sIm {
				t.Fatalf("scatter n=%d m=%d", n, m)
			}
		}
		x32, d32, r32 := to32(x), make([]float64, n), make([]float64, n)
		CMulGatherFrom32(d32, x32, tw)
		cmulGather32Tail(r32, x32, tw, 0)
		for i := range d32 {
			if math.Abs(d32[i]-r32[i]) > 8*u53*(math.Abs(d32[i])+1e-300+float64(n)*math.Abs(tw[i])*16) {
				t.Fatalf("gather32 n=%d i=%d", n, i)
			}
		}
		o32, q32 := make([]float32, n), make([]float32, n)
		CMulScatterTo32(o32, c, tw, 0.75)
		cmulScatter32Ref(q32, c, tw, 0.75)
		for i := range o32 {
			if math.Abs(float64(o32[i])-float64(q32[i])) > 0x1p-23*math.Abs(float64(q32[i]))+1e-3 {
				t.Fatalf("scatter32 n=%d i=%d", n, i)
			}
		}
	})
}

func BenchmarkCMulTwiddle(b *testing.B) {
	rng := rand.New(rand.NewSource(64))
	for _, n := range []int{64, 128, 256, 1024} {
		x, c, tw := twiddleInputs(rng, n)
		dst, x32, d32 := make([]float64, n), to32(x), make([]float32, n)
		name := map[int]string{64: "64", 128: "128", 256: "256", 1024: "1024"}[n]
		b.Run("gather/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CMulGather(dst, x, tw)
			}
		})
		b.Run("gather-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cmulGatherTail(dst, x, tw, 0)
			}
		})
		b.Run("gatherFrom32/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CMulGatherFrom32(dst, x32, tw)
			}
		})
		b.Run("scatter/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CMulScatter(dst, c, tw, 0.5)
			}
		})
		b.Run("scatter-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cmulScatterRef(dst, c, tw, 0.5)
			}
		})
		b.Run("scatterTo32/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CMulScatterTo32(d32, c, tw, 0.5)
			}
		})
	}
}
