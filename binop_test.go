package floats

import (
	"math"
	"math/rand"
	"testing"
)

// The reference of each kernel is the Go expression itself, spelled out here
// (not binopRef): the kernels must agree with it bit for bit on every length,
// alignment and special value (NaN of either sign, signed zeros, infinities,
// subnormals), and must not write past dst.

func TestBinopBitIdentical(t *testing.T) {
	ops := []struct {
		name string
		f    func(dst, a, b []float64)
		ref  func(dst, a, b []float64)
		expr func(x, y float64) float64
	}{
		{"Add", Add, addRef, func(x, y float64) float64 { return x + y }},
		{"Sub", Sub, subRef, func(x, y float64) float64 { return x - y }},
		{"Div", Div, divRef, func(x, y float64) float64 { return x / y }},
	}
	for _, op := range ops {
		forEachCase(t, 31, func(n, off int, sp bool, rng *rand.Rand) {
			a, _ := fillE(n, off, rng, sp)
			b, _ := fillE(n, (off+1)%4, rng, sp)
			dst, buf := fillE(n, (off+2)%4, rng, sp)
			want := make([]float64, n)
			for i := range want {
				want[i] = op.expr(a[i], b[i])
			}
			op.f(dst, a, b)
			checkSame(t, op.name, dst, want, n, off)
			checkGuard(t, op.name, buf, n, (off+2)%4)
			op.ref(dst, a, b)
			checkSame(t, op.name+"Ref", dst, want, n, off)
		})
	}
}

func TestSqrtBitIdentical(t *testing.T) {
	forEachCase(t, 32, func(n, off int, sp bool, rng *rand.Rand) {
		a, _ := fillE(n, off, rng, sp)
		dst, buf := fillE(n, (off+2)%4, rng, sp)
		want := make([]float64, n)
		for i := range want {
			want[i] = math.Sqrt(a[i])
		}
		Sqrt(dst, a)
		checkSame(t, "Sqrt", dst, want, n, off)
		checkGuard(t, "Sqrt", buf, n, (off+2)%4)
		sqrtRef(dst, a)
		checkSame(t, "sqrtRef", dst, want, n, off)
	})
}

func TestClampScalarBitIdentical(t *testing.T) {
	bounds := [][2]float64{{-1, 1}, {0.25, 0.5}, {1, -1}, {math.Inf(-1), 3}, {-3, math.Inf(1)},
		{math.NaN(), 1}, {-1, math.NaN()}, {math.NaN(), math.NaN()}, {negZero, 0}, {0, negZero}}
	forEachCase(t, 33, func(n, off int, sp bool, rng *rand.Rand) {
		for _, bd := range bounds {
			lo, hi := bd[0], bd[1]
			v, _ := fillE(n, off, rng, sp)
			dst, buf := fillE(n, (off+2)%4, rng, sp)
			want := make([]float64, n)
			for i, x := range v { // the documented definition, spelled out
				if lo > x {
					x = lo
				}
				if hi < x {
					x = hi
				}
				want[i] = x
			}
			ClampScalar(dst, v, lo, hi)
			exactBits(t, "ClampScalar", dst, want, n, off)
			checkGuard(t, "ClampScalar", buf, n, (off+2)%4)
			clampScalarRef(dst, v, lo, hi)
			exactBits(t, "clampScalarRef", dst, want, n, off)
		}
	})
}

// ClampScalar with the same bounds at every index is Clamp.
func TestClampScalarIsClamp(t *testing.T) {
	rng := rand.New(rand.NewSource(34))
	for _, n := range []int{0, 1, 5, 64, 129} {
		v, _ := fillE(n, 0, rng, true)
		lo, hi := make([]float64, n), make([]float64, n)
		for i := range lo {
			lo[i], hi[i] = -0.5, 0.75
		}
		a, b := make([]float64, n), make([]float64, n)
		Clamp(a, v, lo, hi)
		ClampScalar(b, v, -0.5, 0.75)
		exactBits(t, "ClampScalar vs Clamp", b, a, n, 0)
	}
}

// dst may be the very slice of an input.
func TestBinopAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(35))
	n := 70
	a, _ := fillE(n, 0, rng, true)
	b, _ := fillE(n, 1, rng, true)
	want := make([]float64, n)
	for _, op := range []struct {
		f    func(dst, a, b []float64)
		expr func(x, y float64) float64
	}{
		{Add, func(x, y float64) float64 { return x + y }},
		{Sub, func(x, y float64) float64 { return x - y }},
		{Div, func(x, y float64) float64 { return x / y }},
	} {
		for i := range want {
			want[i] = op.expr(a[i], b[i])
		}
		x := append([]float64(nil), a...)
		op.f(x, x, b)
		checkSame(t, "alias a", x, want, n, 0)
		y := append([]float64(nil), b...)
		op.f(y, a, y)
		checkSame(t, "alias b", y, want, n, 0)
	}
	x := append([]float64(nil), a...)
	Sqrt(x, x)
	for i := range x {
		want[i] = math.Sqrt(a[i])
	}
	checkSame(t, "Sqrt alias", x, want, n, 0)
	x = append([]float64(nil), a...)
	ClampScalar(x, x, -0.5, 0.5)
	clampScalarRef(want, a, -0.5, 0.5)
	exactBits(t, "ClampScalar alias", x, want, n, 0)
}

// Only the first len(a) elements are touched, and a short operand panics.
func TestBinopLengths(t *testing.T) {
	a, b := []float64{1, 2, 3}, []float64{4, 5, 6, 7, 8}
	dst := []float64{9, 9, 9, 9, 9}
	Add(dst, a, b)
	if dst[0] != 5 || dst[1] != 7 || dst[2] != 9 || dst[3] != 9 || dst[4] != 9 {
		t.Fatalf("Add touched or computed wrongly: %v", dst)
	}
	Sqrt(dst, []float64{4, 9})
	if dst[0] != 2 || dst[1] != 3 || dst[2] != 9 {
		t.Fatalf("Sqrt: %v", dst)
	}
	ClampScalar(dst, []float64{-5, 5}, -1, 1)
	if dst[0] != -1 || dst[1] != 1 || dst[2] != 9 {
		t.Fatalf("ClampScalar: %v", dst)
	}
	short := []float64{1, 2}
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		f()
	}
	mustPanic("Add b", func() { Add(dst, a, short) })
	mustPanic("Add dst", func() { Add(short, a, b) })
	mustPanic("Sub b", func() { Sub(dst, a, short) })
	mustPanic("Sub dst", func() { Sub(short, a, b) })
	mustPanic("Div b", func() { Div(dst, a, short) })
	mustPanic("Div dst", func() { Div(short, a, b) })
	mustPanic("Sqrt dst", func() { Sqrt(short, a) })
	mustPanic("ClampScalar dst", func() { ClampScalar(short, a, 0, 1) })
}
