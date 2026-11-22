package floats

import (
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// mulRows32Inputs fills a matrix with finite values of several magnitudes and signs.
func mulRows32Inputs(rng *rand.Rand, n int) []float32 {
	c := make([]float32, n)
	for i := range c {
		switch rng.Intn(16) {
		case 0:
			c[i] = 0
		case 1:
			c[i] = float32(math.Copysign(0, -1))
		default:
			c[i] = float32(rng.NormFloat64() * math.Pow(10, float64(rng.Intn(7)-3)))
		}
	}
	return c
}

// mulRows32Bound is the bound of the contract for element j: rows * 2^-24 * (|d0| +
// Σ|a*b|) * (1 + rows * 2^-24), computed in float64.
func mulRows32Bound(d0 float32, a []float32, lda int, b []float32, ldb int, rows, j int) float64 {
	mag := math.Abs(float64(d0))
	for k := 0; k < rows; k++ {
		mag += math.Abs(float64(a[k*lda+j]) * float64(b[k*ldb+j]))
	}
	r := float64(rows) * 0x1p-24
	return r * mag * (1 + r)
}

// mulRows32Exact is dst + Σ a*b in 300 bits.
func mulRows32Exact(d0 float32, a []float32, lda int, b []float32, ldb int, rows, j int) *big.Float {
	exact := new(big.Float).SetPrec(300).SetFloat64(float64(d0))
	for k := 0; k < rows; k++ {
		p := new(big.Float).SetPrec(300).SetFloat64(float64(a[k*lda+j]))
		p.Mul(p, new(big.Float).SetPrec(300).SetFloat64(float64(b[k*ldb+j])))
		exact.Add(exact, p)
	}
	return exact
}

func mulRows32Err(got float32, exact *big.Float) float64 {
	diff := new(big.Float).SetPrec(300).Sub(new(big.Float).SetPrec(300).SetFloat64(float64(got)), exact)
	e, _ := diff.Float64()
	return math.Abs(e)
}

// TestMulAddRows32AgainstExact asserts the stated bound of MulAddRows32 against a 300-bit
// evaluation, for the public function and the portable loop, at every shape of the
// float64 tests.
func TestMulAddRows32AgainstExact(t *testing.T) {
	rng := rand.New(rand.NewSource(301))
	var worst float64
	for _, s := range mulRowsShapes() {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := mulRows32Inputs(rng, (rows-1)*lda+n)
		b := mulRows32Inputs(rng, (rows-1)*ldb+n)
		d0 := mulRows32Inputs(rng, n)
		got := append([]float32(nil), d0...)
		MulAddRows32(got, a, lda, b, ldb, rows)
		ref := append([]float32(nil), d0...)
		mulAddRows32Ref(ref, a, lda, b, ldb, rows)
		for j := 0; j < n; j++ {
			bound := mulRows32Bound(d0[j], a, lda, b, ldb, rows, j)
			exact := mulRows32Exact(d0[j], a, lda, b, ldb, rows, j)
			for name, v := range map[string]float32{"MulAddRows32": got[j], "portable loop": ref[j]} {
				e := mulRows32Err(v, exact)
				if bound > 0 {
					worst = max(worst, e/bound)
				}
				if e > bound {
					t.Fatalf("%s n=%d rows=%d lda=%d ldb=%d element %d: error %g, bound %g", name, n, rows, lda, ldb, j, e, bound)
				}
			}
		}
	}
	t.Logf("worst error %.3f of the bound", worst)
}

// TestMulAddRows32NonFinite: infinities and NaNs propagate as in the portable loop.
func TestMulAddRows32NonFinite(t *testing.T) {
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	a := []float32{1, inf, 2, 3, nan, 5, 6, 7, 8, 9}
	b := []float32{1, 1, 0, 1, 1, 1, 1, 1, 1, 1}
	for n := 1; n <= len(a); n++ {
		got, ref := make([]float32, n), make([]float32, n)
		MulAddRows32(got, a, n, b, n, 1)
		mulAddRows32Ref(ref, a, n, b, n, 1)
		for j := range got {
			if got[j] != ref[j] && !(got[j] != got[j] && ref[j] != ref[j]) {
				t.Fatalf("n=%d element %d: %v, portable %v", n, j, got[j], ref[j])
			}
		}
	}
}

func TestMulAddRows32PanicsAndBounds(t *testing.T) {
	// The message is asserted: a runtime index panic from a deleted guard is a
	// panic too, and would pass for the refusal.
	mustPanic := func(name, msg string, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("%s: no panic", name)
				return
			}
			if s, ok := r.(string); !ok || !strings.Contains(s, msg) {
				t.Errorf("%s: panic %v, want a message containing %q", name, r, msg)
			}
		}()
		f()
	}
	d, a, b := make([]float32, 4), make([]float32, 8), make([]float32, 8)
	mustPanic("rows<0", "rows < 0", func() { MulAddRows32(d, a, 4, b, 4, -1) })
	mustPanic("lda<0", "lda or ldb < 0", func() { MulAddRows32(d, a, -1, b, 4, 1) })
	mustPanic("ldb<0", "lda or ldb < 0", func() { MulAddRows32(d, a, 4, b, -1, 1) })
	mustPanic("short a", "shorter", func() { MulAddRows32(d, a[:7], 4, b, 4, 2) })
	mustPanic("short b", "shorter", func() { MulAddRows32(d, a, 4, b[:7], 4, 2) })
	mustPanic("overflow", "overflows", func() { MulAddRows32(d, a, math.MaxInt, b, 4, 3) })
	mustPanic("dst overlaps a", "overlaps", func() { MulAddRows32(a[:4], a, 4, b, 4, 2) })
	mustPanic("dst overlaps b", "overlaps", func() { MulAddRows32(b[4:], a, 4, b, 4, 2) })
}

// TestMulAddRows32Empty: no rows or no columns leave dst as it was and read nothing.
func TestMulAddRows32Empty(t *testing.T) {
	d := []float32{1, 2, 3}
	// quiet: an empty call must return, not panic; a panic is reported as an assertion.
	quiet := func(name string, f func()) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("%s: panicked on an empty call: %v", name, r)
			}
		}()
		f()
	}
	quiet("rows=0", func() { MulAddRows32(d, nil, 0, nil, 0, 0) })
	if d[0] != 1 || d[1] != 2 || d[2] != 3 {
		t.Errorf("rows = 0 changed dst: %v", d)
	}
	quiet("n=0 nil dst", func() { MulAddRows32(nil, nil, 0, nil, 0, 5) })
	quiet("n=0 stride", func() { MulAddRows32(d[:0], nil, 7, nil, 7, 5) })
	MulAddRows32(d, []float32{2, 2, 2}, 0, []float32{3, 3, 3}, 0, 1)
	if d[0] != 7 || d[1] != 8 || d[2] != 9 {
		t.Errorf("one row: %v, want [7 8 9]", d)
	}
}

// FuzzMulAddRows32: for any shape and any bytes (finite values), the public function
// stays within the bound of the portable loop's own bound of the exact value.
func FuzzMulAddRows32(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), []byte{1, 2, 3})
	f.Add(uint8(63), uint8(9), uint8(0), uint8(64), []byte("window sum"))
	f.Add(uint8(33), uint8(3), uint8(5), uint8(5), []byte{255, 0, 128})
	f.Fuzz(func(t *testing.T, n8, rows8, pa, pb uint8, data []byte) {
		n, rows := int(n8)%80, int(rows8)%12
		lda, ldb := n+int(pa)%70, n+int(pb)%70
		if pa%7 == 0 {
			lda = 0
		}
		fill := func(m int, salt int) []float32 {
			v := make([]float32, m)
			for i := range v {
				if len(data) > 0 {
					x := math.Float32frombits(uint32(data[(i*3+salt)%len(data)])<<24 | uint32(data[(i+salt)%len(data)])<<16 | uint32(i))
					if x-x != 0 { // not finite
						x = 1
					}
					v[i] = x
				}
			}
			return v
		}
		a, b, d := fill(max(rows-1, 0)*lda+n, 1), fill(max(rows-1, 0)*ldb+n, 2), fill(n, 3)
		d0 := append([]float32(nil), d...)
		MulAddRows32(d, a, lda, b, ldb, rows)
		for j := 0; j < n; j++ {
			bound := mulRows32Bound(d0[j], a, lda, b, ldb, rows, j)
			if e := mulRows32Err(d[j], mulRows32Exact(d0[j], a, lda, b, ldb, rows, j)); e > bound && !math.IsInf(float64(d[j]), 0) {
				t.Fatalf("n=%d rows=%d element %d: error %g, bound %g", n, rows, j, e, bound)
			}
		}
	})
}

// BenchmarkMulAddRows32 is the window sum of a 64-band QMF synthesis in float32.
func BenchmarkMulAddRows32(b *testing.B) {
	rng := rand.New(rand.NewSource(304))
	win := mulRows32Inputs(rng, 5*128)
	v := mulRows32Inputs(rng, 5*256)
	d := make([]float32, 64)
	b.Run("kernel/64x5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MulAddRows32(d, win, 128, v, 256, 5)
		}
	})
	b.Run("portable/64x5", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			mulAddRows32Ref(d, win, 128, v, 256, 5)
		}
	})
}
