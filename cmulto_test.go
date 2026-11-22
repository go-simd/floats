package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// cmulOracle is a*b in 300-bit arithmetic, rounded to float64, with the scale |a|·|b|.
func cmulOracle(a, b complex128) (re, im, scale float64) {
	f := func(v float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(v) }
	mul := func(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(300).Mul(x, y) }
	ar, ai, br, bi := f(real(a)), f(imag(a)), f(real(b)), f(imag(b))
	re, _ = new(big.Float).SetPrec(300).Sub(mul(ar, br), mul(ai, bi)).Float64()
	im, _ = new(big.Float).SetPrec(300).Add(mul(ar, bi), mul(ai, br)).Float64()
	scale = math.Hypot(real(a), imag(a)) * math.Hypot(real(b), imag(b))
	return
}

// cmulTol is the stated bound: 2 eps (2^-52) of |a|·|b| plus an absolute floor for the
// products that underflow to subnormals.
func cmulTol(scale float64) float64 {
	return 2*0x1p-52*scale + 64*math.SmallestNonzeroFloat64
}

func cmulOK(got complex128, a, b complex128) bool {
	re, im, sc := cmulOracle(a, b)
	tol := cmulTol(sc)
	return math.Abs(real(got)-re) <= tol && math.Abs(imag(got)-im) <= tol
}

func cmulRand(rng *rand.Rand, n int, scale float64) []complex128 {
	v := make([]complex128, n)
	for i := range v {
		v[i] = complex(rng.NormFloat64()*scale, rng.NormFloat64()*scale)
	}
	return v
}

func cmulLens() []int {
	var l []int
	for n := 0; n <= 70; n++ {
		l = append(l, n)
	}
	return append(l, 127, 128, 129, 255, 1000, 4097)
}

func checkCMulTo(t *testing.T) {
	t.Helper()
	rng := rand.New(rand.NewSource(5))
	sentinel := complex(mirrorSentinel, -mirrorSentinel)
	for _, n := range cmulLens() {
		a, b := cmulRand(rng, n, 4), cmulRand(rng, n, 0.01)
		dst := make([]complex128, n+3)
		for i := range dst {
			dst[i] = sentinel
		}
		CMulTo(dst[:n], a, b)
		for i := 0; i < n; i++ {
			if !cmulOK(dst[i], a[i], b[i]) {
				t.Fatalf("n=%d i=%d: got %v for %v*%v", n, i, dst[i], a[i], b[i])
			}
		}
		for i := n; i < len(dst); i++ {
			if dst[i] != sentinel {
				t.Fatalf("n=%d: wrote past n at %d", n, i)
			}
		}
	}
}

func TestCMulToOracle(t *testing.T) { checkCMulTo(t) }

// TestCMulToLongerOperands: a and b longer than dst are read only up to len(dst).
func TestCMulToLongerOperands(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	a, b := cmulRand(rng, 20, 1), cmulRand(rng, 20, 1)
	dst := make([]complex128, 9)
	CMulTo(dst, a, b)
	for i := range dst {
		if !cmulOK(dst[i], a[i], b[i]) {
			t.Fatalf("i=%d: got %v", i, dst[i])
		}
	}
}

// TestCMulToInPlace: dst may be a or b exactly, at every length around the kernel's step.
func TestCMulToInPlace(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range cmulLens() {
		a0, b0 := cmulRand(rng, n, 2), cmulRand(rng, n, 2)
		a := append([]complex128(nil), a0...)
		CMulTo(a, a, b0)
		for i := range a {
			if !cmulOK(a[i], a0[i], b0[i]) {
				t.Fatalf("dst==a n=%d i=%d: got %v", n, i, a[i])
			}
		}
		b := append([]complex128(nil), b0...)
		CMulTo(b, a0, b)
		for i := range b {
			if !cmulOK(b[i], a0[i], b0[i]) {
				t.Fatalf("dst==b n=%d i=%d: got %v", n, i, b[i])
			}
		}
		c := append([]complex128(nil), a0...)
		CMulTo(c, c, c) // a == b == dst: the square
		for i := range c {
			if !cmulOK(c[i], a0[i], a0[i]) {
				t.Fatalf("dst==a==b n=%d i=%d: got %v", n, i, c[i])
			}
		}
	}
}

// TestCMulToValues: exact small values, which a swapped part or a wrong sign shows at once.
func TestCMulToValues(t *testing.T) {
	a := []complex128{1 + 2i, 3 - 1i, 0 + 1i, -2 - 2i, 5, 1i, 2 + 0i, -1 + 1i, 1, 2}
	b := []complex128{3 + 4i, 1 + 1i, 0 + 1i, 2 - 3i, 1i, 5, 0.5 + 0.5i, 1 - 1i, 7, 8}
	dst := make([]complex128, len(a))
	CMulTo(dst, a, b)
	for i := range dst {
		if want := a[i] * b[i]; dst[i] != want {
			t.Errorf("i=%d: %v * %v = %v, want %v", i, a[i], b[i], dst[i], want)
		}
	}
}

func TestCMulToPanics(t *testing.T) {
	sentinel := complex(mirrorSentinel, 0)
	for _, c := range []struct {
		name string
		n    int
		la   int
		lb   int
	}{
		{"short a", 8, 7, 8},
		{"short b", 8, 8, 7},
		{"both short", 8, 0, 3},
		{"empty a", 1, 0, 1},
	} {
		dst := make([]complex128, c.n)
		for i := range dst {
			dst[i] = sentinel
		}
		a, b := make([]complex128, c.la, c.n+4), make([]complex128, c.lb, c.n+4) // room in the capacity
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", c.name)
				}
			}()
			CMulTo(dst, a, b)
		}()
		for i, v := range dst {
			if v != sentinel {
				t.Errorf("%s: dst[%d] written before the panic", c.name, i)
			}
		}
	}
	CMulTo(nil, nil, nil) // empty: nothing to do
	CMulTo(make([]complex128, 0, 4), nil, nil)
}

// TestCMulToKeepsSlicesOnStack: the kernel stubs are go:noescape, so the caller's arrays
// stay on its stack and a call allocates nothing.
func TestCMulToKeepsSlicesOnStack(t *testing.T) {
	allocs := testing.AllocsPerRun(20, func() {
		var a, b, d [37]complex128
		for i := range a {
			a[i], b[i] = complex(float64(i), 1), complex(1, float64(i))
		}
		CMulTo(d[:], a[:], b[:])
		CMulTo(a[:], a[:], b[:])
		if d[3] == 0 {
			panic("unreachable")
		}
	})
	if allocs != 0 {
		t.Errorf("%v heap allocations per call: a kernel stub lets its slices escape", allocs)
	}
}

func FuzzCMulTo(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, uint8(3))
	f.Add(make([]byte, 8*4*20), uint8(9))
	f.Fuzz(func(t *testing.T, data []byte, nn uint8) {
		v := bytesToF64(data)
		n := int(nn) % 50
		if len(v) < 4*n {
			return
		}
		a, b := make([]complex128, n), make([]complex128, n)
		for i := 0; i < n; i++ {
			a[i], b[i] = complex(v[4*i], v[4*i+1]), complex(v[4*i+2], v[4*i+3])
		}
		dst := make([]complex128, n)
		CMulTo(dst, a, b)
		for i := range dst {
			if !cmulOK(dst[i], a[i], b[i]) {
				re, im, sc := cmulOracle(a[i], b[i])
				t.Fatalf("n=%d i=%d: %v*%v = %v, oracle (%v,%v) tol %v", n, i, a[i], b[i], dst[i], re, im, cmulTol(sc))
			}
		}
	})
}
