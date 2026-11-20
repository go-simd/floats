package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// zipRevOracle is dst[j] in 300-bit arithmetic with its scale |w|·|(x, y')|.
func zipRevOracle(j, m int, x, y, w []float64) (re, im, scale float64) {
	f := func(v float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(v) }
	mul := func(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(300).Mul(a, b) }
	wr, wi, a, b := f(w[2*j]), f(w[2*j+1]), f(x[j]), f(y[m-1-j])
	re, _ = new(big.Float).SetPrec(300).Add(mul(wr, a), mul(wi, b)).Float64()
	im, _ = new(big.Float).SetPrec(300).Sub(mul(wi, a), mul(wr, b)).Float64()
	scale = math.Hypot(w[2*j], w[2*j+1]) * math.Hypot(x[j], y[m-1-j])
	return
}

// zipRevTol is the bound: 4 eps of the scale plus an absolute floor for the products that
// underflow to subnormals (found by -fuzz, testdata/fuzz).
func zipRevTol(scale float64) float64 {
	return 4*0x1p-52*scale + 64*math.SmallestNonzeroFloat64
}

func checkZipRev(t *testing.T, lens []int) {
	t.Helper()
	rng := rand.New(rand.NewSource(3))
	for _, m := range lens {
		x, y, w := make([]float64, m), make([]float64, m), make([]float64, 2*m)
		for i := range x {
			x[i], y[i] = rng.NormFloat64()*4, rng.NormFloat64()*4
		}
		for i := range w {
			w[i] = rng.NormFloat64()
		}
		dst := make([]float64, 2*m+3)
		for i := range dst {
			dst[i] = mirrorSentinel
		}
		CMulZipRev(dst[:2*m], x, y, w)
		for j := 0; j < m; j++ {
			re, im, sc := zipRevOracle(j, m, x, y, w)
			if math.Abs(dst[2*j]-re) > zipRevTol(sc) || math.Abs(dst[2*j+1]-im) > zipRevTol(sc) {
				t.Fatalf("m=%d j=%d: got (%v,%v) want (%v,%v)", m, j, dst[2*j], dst[2*j+1], re, im)
			}
		}
		for i := 2 * m; i < len(dst); i++ {
			if dst[i] != mirrorSentinel {
				t.Fatalf("m=%d: wrote past 2m at %d", m, i)
			}
		}
	}
}

func zipRevLens() []int {
	var l []int
	for m := 0; m <= 130; m++ {
		l = append(l, m)
	}
	return append(l, 255, 256, 257, 1000, 4097)
}

func TestCMulZipRevOracle(t *testing.T) { checkZipRev(t, zipRevLens()) }

// TestCMulZipRevMirrorsY: y is read from its far end (a kernel that reads it forward is
// told from the oracle), and x and y may be the two halves of one slice.
func TestCMulZipRevMirrorsY(t *testing.T) {
	m := 8
	src := make([]float64, 2*m)
	for i := range src {
		src[i] = float64(i + 1)
	}
	w := make([]float64, 2*m)
	for j := 0; j < m; j++ {
		w[2*j] = 1 // w = 1: dst[j] = (x[j], -y[m-1-j])
	}
	dst := make([]float64, 2*m)
	CMulZipRev(dst, src[:m], src[m:], w)
	for j := 0; j < m; j++ {
		if dst[2*j] != src[j] || dst[2*j+1] != -src[m+m-1-j] {
			t.Fatalf("j=%d: got (%v,%v) want (%v,%v)", j, dst[2*j], dst[2*j+1], src[j], -src[2*m-1-j])
		}
	}
}

func TestCMulZipRevPanics(t *testing.T) {
	ok := func() (dst, x, y, w []float64) {
		return make([]float64, 8), make([]float64, 4), make([]float64, 4), make([]float64, 8)
	}
	cases := []struct {
		name string
		cut  func(dst, x, y, w []float64) (a, b, c, d []float64)
	}{
		{"odd dst", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst[:7], x, y, w }},
		{"short x", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst, x[:3], y, w }},
		{"short y", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst, x, y[:3], w }},
		{"short w", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst, x, y, w[:7] }},
		{"dst is w", func(dst, x, y, w []float64) (a, b, c, d []float64) { return w, x, y, w }},
		{"dst holds x", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst, dst[2:6], y, w }},
		{"dst holds y", func(dst, x, y, w []float64) (a, b, c, d []float64) { return dst, x, dst[4:8], w }},
	}
	for _, c := range cases {
		dst, x, y, w := ok()
		for i := range dst {
			dst[i], w[i] = mirrorSentinel, 1
		}
		a, b, cc, d := c.cut(dst, x, y, w)
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", c.name)
				}
			}()
			CMulZipRev(a, b, cc, d)
		}()
		if c.name == "dst is w" || c.name == "dst holds x" || c.name == "dst holds y" {
			continue
		}
		for i, v := range dst {
			if v != mirrorSentinel {
				t.Errorf("%s: dst[%d] written before the panic", c.name, i)
			}
		}
	}
	CMulZipRev(nil, nil, nil, nil) // empty: nothing to do
}

func FuzzCMulZipRev(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, uint8(3))
	f.Add(make([]byte, 8*40), uint8(9))
	f.Fuzz(func(t *testing.T, data []byte, mm uint8) {
		a := bytesToF64(data)
		m := int(mm) % 70
		if len(a) < 4*m {
			return
		}
		x, y, w := a[:m], a[m:2*m], a[2*m:4*m]
		dst := make([]float64, 2*m)
		CMulZipRev(dst, x, y, w)
		for j := 0; j < m; j++ {
			re, im, sc := zipRevOracle(j, m, x, y, w)
			if tol := zipRevTol(sc); math.Abs(dst[2*j]-re) > tol || math.Abs(dst[2*j+1]-im) > tol {
				t.Fatalf("m=%d j=%d: kernel (%v,%v) oracle (%v,%v) tol %v", m, j, dst[2*j], dst[2*j+1], re, im, tol)
			}
		}
	})
}
