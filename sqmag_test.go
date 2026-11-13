package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
	"unsafe"
)

func sqmagLengths() []int {
	var l []int
	for n := 1; n <= 70; n++ {
		l = append(l, n)
	}
	return append(l, 127, 128, 129, 255, 256, 257, 1000, 1024, 4097)
}

func sqmagInputs(rng *rand.Rand, n int) []float64 {
	c := make([]float64, 2*n)
	for i := range c {
		c[i] = rng.NormFloat64() * 100
	}
	return c
}

// sqmagOracle is the exact (re²+im²)·s in 200 bits.
func sqmagOracle(re, im, s float64) *big.Float {
	a := new(big.Float).SetPrec(overlapPrec).Mul(bigOf(re), bigOf(re))
	b := new(big.Float).SetPrec(overlapPrec).Mul(bigOf(im), bigOf(im))
	a.Add(a, b)
	return a.Mul(a, bigOf(s))
}

// TestSquaredMagnitudeBound: every output within 4 units of the last place of the
// exact value, for the kernel and for the portable loop, at every length from 1 to 70
// and at the sizes of a spectrogram.
func TestSquaredMagnitudeBound(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	for _, n := range sqmagLengths() {
		c := sqmagInputs(rng, n)
		for _, s := range []float64{1, 0.001 / 256, 2 * 0.001 / 256, -3.5} {
			got, ref := make([]float64, n), make([]float64, n)
			SquaredMagnitude(got, c, s)
			sqmagTail(ref, c, s, 0)
			for name, d := range map[string][]float64{"SquaredMagnitude": got, "sqmagTail": ref} {
				for k := 0; k < n; k++ {
					exact := sqmagOracle(c[2*k], c[2*k+1], s)
					diff := new(big.Float).SetPrec(overlapPrec).Sub(bigOf(d[k]), exact)
					e, _ := diff.Float64()
					ev, _ := exact.Float64()
					if math.Abs(e) > 4*u53*math.Abs(ev) {
						t.Fatalf("%s n=%d s=%g bin %d: %g, exact %g: error %g exceeds %g", name, n, s, k, d[k], ev, e, 4*u53*math.Abs(ev))
					}
				}
			}
		}
	}
}

func TestSquaredMagnitudePanicsAndBounds(t *testing.T) {
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
	mustPanic("short c", "floats: SquaredMagnitude: len(c) < 2*len(dst)", func() { SquaredMagnitude(make([]float64, 4), make([]float64, 7), 1) })
	SquaredMagnitude(nil, nil, 1) // n = 0: nothing
	buf := make([]float64, 64)
	mustPanic("dst == c", "floats: SquaredMagnitude: dst overlaps c", func() { SquaredMagnitude(buf[:8], buf[:16], 1) })
	mustPanic("shifted", "floats: SquaredMagnitude: dst overlaps c", func() { SquaredMagnitude(buf[12:20], buf[:16], 1) })
	mustPanic("c in dst", "floats: SquaredMagnitude: dst overlaps c", func() { SquaredMagnitude(buf[:16], buf[4:36], 1) })
	mustPanic("across element types", "floats: SquaredMagnitude: dst overlaps c", func() {
		SquaredMagnitude(unsafe.Slice((*float64)(unsafe.Pointer(&buf[8])), 8), buf[:16], 1)
	})
	// adjacent slices of one array do not overlap
	rng := rand.New(rand.NewSource(92))
	c := sqmagInputs(rng, 8)
	copy(buf[:16], c)
	SquaredMagnitude(buf[16:24], buf[:16], 0.5)
	for k := 0; k < 8; k++ {
		want := float64(float64(c[2*k]*c[2*k])+float64(c[2*k+1]*c[2*k+1])) * 0.5
		if math.Abs(buf[16+k]-want) > 8*u53*want {
			t.Fatalf("adjacent: bin %d = %g, want %g", k, buf[16+k], want)
		}
	}
	// nothing written outside dst[:n], nothing read past 2n, c unmodified
	for _, n := range sqmagLengths() {
		c := sqmagInputs(rng, n)
		c0 := append([]float64(nil), c...)
		long := append(append([]float64(nil), c...), 1e300, 1e300, 1e300, 1e300)
		pad := make([]float64, n+16)
		for i := range pad {
			pad[i] = 12345
		}
		SquaredMagnitude(pad[8:8+n], long, 1e-3)
		for i := range pad {
			if (i < 8 || i >= 8+n) && pad[i] != 12345 {
				t.Fatalf("n=%d: wrote outside dst at %d", n, i)
			}
		}
		for i := range c {
			if c[i] != c0[i] {
				t.Fatalf("n=%d: c changed", n)
			}
		}
		for i := 8; i < 8+n; i++ {
			if pad[i] > 1e8 || math.IsNaN(pad[i]) {
				t.Fatalf("n=%d: an operand past 2n was read (%g)", n, pad[i])
			}
		}
	}
}

// FuzzSquaredMagnitude: for any length 1 to 130 and any bytes, the kernel and the
// portable loop agree to the bound of the contract.
func FuzzSquaredMagnitude(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3})
	f.Add(uint8(1), []byte("squared magnitude"))
	f.Add(uint8(3), []byte{255, 0, 128})
	f.Add(uint8(64), []byte{})
	f.Fuzz(func(t *testing.T, k uint8, data []byte) {
		n := int(k)%130 + 1
		c := make([]float64, 2*n)
		for i := range c {
			if len(data) > 0 {
				c[i] = float64(int(data[(i*3+1)%len(data)])-128) / 8
			}
		}
		got, ref := make([]float64, n), make([]float64, n)
		SquaredMagnitude(got, c, 0.75)
		sqmagTail(ref, c, 0.75, 0)
		for i := range got {
			if math.Abs(got[i]-ref[i]) > 8*u53*math.Abs(ref[i]) {
				t.Fatalf("n=%d bin %d: %g, portable %g", n, i, got[i], ref[i])
			}
		}
	})
}

func BenchmarkSquaredMagnitude(b *testing.B) {
	rng := rand.New(rand.NewSource(93))
	for _, n := range []int{129, 1025} { // the bins of a 256 and a 2,048 point real transform
		c := sqmagInputs(rng, n)
		d := make([]float64, n)
		name := map[int]string{129: "129", 1025: "1025"}[n]
		b.Run("kernel/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				SquaredMagnitude(d, c, 0.5)
			}
		})
		b.Run("portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sqmagTail(d, c, 0.5, 0)
			}
		})
	}
}
