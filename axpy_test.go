package floats

import (
	"math"
	"math/rand"
	"testing"
)

// sameBits reports whether a and b are the same float64 bit for bit, treating
// any two NaNs as equal (IEEE 754 leaves the payload of a NaN result to the
// hardware; NaN-ness is what the contract promises).
func sameBits(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// specials are planted into the random inputs to check IEEE propagation: the
// fused kernels must produce exactly what math.FMA produces for them too.
var specials = []float64{
	math.NaN(), math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1),
	math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, 1e-310, 1, -1,
}

// fill returns n random values carved from a backing slice at element offset
// off, followed by guard elements holding a sentinel, so the tests see both an
// unaligned start and any write past the end of the slice.
const guard = 5

var sentinel = math.Float64frombits(0x7ff8dead_beefcafe)

func fill(n, off int, rng *rand.Rand, withSpecials bool) (s, buf []float64) {
	buf = make([]float64, off+n+guard)
	for i := range buf {
		buf[i] = sentinel
	}
	s = buf[off : off+n]
	for i := range s {
		s[i] = rng.NormFloat64() * math.Exp2(float64(rng.Intn(20)-10))
		if withSpecials && rng.Intn(8) == 0 {
			s[i] = specials[rng.Intn(len(specials))]
		}
	}
	return s, buf
}

func checkGuard(t *testing.T, what string, buf []float64, n, off int) {
	t.Helper()
	for i := off + n; i < len(buf); i++ {
		if math.Float64bits(buf[i]) != math.Float64bits(sentinel) {
			t.Fatalf("%s n=%d off=%d: wrote past the end at %d", what, n, off, i-off)
		}
	}
}

func checkSame(t *testing.T, what string, got, want []float64, n, off int) {
	t.Helper()
	for i := range want {
		if !sameBits(got[i], want[i]) {
			t.Fatalf("%s n=%d off=%d: [%d] = %v (%#x), want %v (%#x)", what, n, off, i,
				got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

// forEachCase runs body over every length 0..257, element offsets 0..3 (so the
// vector loads start unaligned) and with or without planted IEEE specials.
func forEachCase(t *testing.T, seed int64, body func(n, off int, specials bool, rng *rand.Rand)) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	for n := 0; n <= 257; n++ {
		for off := 0; off < 4; off++ {
			for _, sp := range []bool{false, true} {
				body(n, off, sp, rng)
			}
		}
	}
}

func alphas(rng *rand.Rand) []float64 {
	return []float64{0, 1, -1, 0.5, rng.NormFloat64(), math.NaN(), math.Inf(-1)}
}

func TestAxpyBitIdentical(t *testing.T) {
	forEachCase(t, 11, func(n, off int, sp bool, rng *rand.Rand) {
		for _, a := range alphas(rng) {
			x, _ := fill(n, off, rng, sp)
			y, buf := fill(n, (off+1)%4, rng, sp)
			want := append([]float64(nil), y...)
			axpyRef(a, x, want)
			Axpy(a, x, y)
			checkSame(t, "Axpy", y, want, n, off)
			checkGuard(t, "Axpy", buf, n, (off+1)%4)
		}
	})
}

func TestXpbyBitIdentical(t *testing.T) {
	forEachCase(t, 12, func(n, off int, sp bool, rng *rand.Rand) {
		for _, b := range alphas(rng) {
			x, _ := fill(n, off, rng, sp)
			y, buf := fill(n, (off+3)%4, rng, sp)
			want := append([]float64(nil), y...)
			xpbyRef(b, x, want)
			Xpby(x, b, y)
			checkSame(t, "Xpby", y, want, n, off)
			checkGuard(t, "Xpby", buf, n, (off+3)%4)
		}
	})
}

func TestScaleBitIdentical(t *testing.T) {
	forEachCase(t, 13, func(n, off int, sp bool, rng *rand.Rand) {
		for _, a := range alphas(rng) {
			x, buf := fill(n, off, rng, sp)
			want := append([]float64(nil), x...)
			scaleRef(a, want)
			Scale(a, x)
			checkSame(t, "Scale", x, want, n, off)
			checkGuard(t, "Scale", buf, n, off)
		}
	})
}

func TestAxpyDot(t *testing.T) {
	forEachCase(t, 14, func(n, off int, sp bool, rng *rand.Rand) {
		for _, a := range alphas(rng) {
			x, _ := fill(n, off, rng, sp)
			y, buf := fill(n, (off+2)%4, rng, sp)
			w, _ := fill(n, (off+1)%4, rng, sp)
			want := append([]float64(nil), y...)
			wantDot := axpyDotRef(a, x, want, w)
			got := AxpyDot(a, x, y, w)
			checkSame(t, "AxpyDot y", y, want, n, off)
			checkGuard(t, "AxpyDot", buf, n, (off+2)%4)
			if math.IsInf(wantDot, 0) || math.IsNaN(wantDot) {
				if !sameBits(got, wantDot) && !(math.IsNaN(got) || math.IsInf(got, 0)) {
					t.Fatalf("AxpyDot n=%d off=%d: got %v want non-finite %v", n, off, got, wantDot)
				}
				continue
			}
			if !closeDot(got, wantDot, absSumProd64(want, w), relTol64) {
				t.Fatalf("AxpyDot n=%d off=%d alpha=%v: got %v want %v", n, off, a, got, wantDot)
			}
		}
	})
}

// TestAxpyDotMatchesAxpyThenDot pins AxpyDot to the two-pass definition it
// fuses: its y equals Axpy's bit for bit and its sum equals Dot's to within
// the reduction tolerance.
func TestAxpyDotMatchesAxpyThenDot(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	for _, n := range sizes {
		x, y, w := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		y2 := append([]float64(nil), y...)
		got := AxpyDot(0.75, x, y, w)
		Axpy(0.75, x, y2)
		checkSame(t, "AxpyDot vs Axpy", y, y2, n, 0)
		if want := Dot(y2, w); !closeDot(got, want, absSumProd64(y2, w), relTol64) {
			t.Errorf("AxpyDot n=%d: %v, Dot after Axpy %v", n, got, want)
		}
	}
}

// TestAxpyAliasing covers the in-place forms the solvers use: y == x, and
// AxpyDot(-α, Ap, r, r) where the dot partner is the updated slice itself.
func TestAxpyAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	for n := 0; n <= 257; n++ {
		x := randF64(n, rng)
		want := append([]float64(nil), x...)
		for i := range want {
			want[i] = math.FMA(1.5, want[i], want[i])
		}
		Axpy(1.5, x, x)
		checkSame(t, "Axpy(x,x)", x, want, n, 0)

		for i := range want {
			want[i] = math.FMA(-0.25, want[i], want[i])
		}
		Xpby(x, -0.25, x)
		checkSame(t, "Xpby(x,x)", x, want, n, 0)

		ap, r := randF64(n, rng), randF64(n, rng)
		rw := append([]float64(nil), r...)
		var ss float64
		for i := range rw {
			rw[i] = math.FMA(-0.5, ap[i], rw[i])
			ss += rw[i] * rw[i]
		}
		got := AxpyDot(-0.5, ap, r, r)
		checkSame(t, "AxpyDot(r,r) y", r, rw, n, 0)
		if !closeDot(got, ss, absSumProd64(rw, rw), relTol64) {
			t.Fatalf("AxpyDot(r,r) n=%d: %v want %v", n, got, ss)
		}
	}
}

// TestAxpyLongerY checks that only the first len(x) elements of a longer y (and
// w) take part.
func TestAxpyLongerY(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{10, 20, 30, 40, 50}
	Axpy(2, x, y)
	if want := []float64{12, 24, 36, 40, 50}; !equalF64(y, want) {
		t.Errorf("Axpy = %v want %v", y, want)
	}
	Xpby(x, 0.5, y)
	if want := []float64{7, 14, 21, 40, 50}; !equalF64(y, want) {
		t.Errorf("Xpby = %v want %v", y, want)
	}
	w := []float64{1, 1, 1, 1000}
	if got := AxpyDot(-1, x, y, w); got != 6+12+18 {
		t.Errorf("AxpyDot = %v want 36", got)
	}
	if want := []float64{6, 12, 18, 40, 50}; !equalF64(y, want) {
		t.Errorf("AxpyDot y = %v want %v", y, want)
	}
	Scale(-2, y)
	if want := []float64{-12, -24, -36, -80, -100}; !equalF64(y, want) {
		t.Errorf("Scale = %v want %v", y, want)
	}
}

func equalF64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestAxpyFusedNotUnfused shows the kernels round once: with a = 1+2⁻³⁰ and
// x = 1−2⁻³⁰, a·x = 1−2⁻⁶⁰ rounds to 1 on its own, so an unfused y + a·x with
// y = −1 gives 0, while the fused form keeps −2⁻⁶⁰.
func TestAxpyFusedNotUnfused(t *testing.T) {
	a := 1 + math.Exp2(-30)
	x := []float64{1 - math.Exp2(-30)} // a*x = 1 - 2⁻⁶⁰, inexact in float64
	y := []float64{-1}
	Axpy(a, x, y)
	if want := -math.Exp2(-60); y[0] != want {
		t.Errorf("Axpy fused = %v want %v", y[0], want)
	}
}

func TestAxpyPanics(t *testing.T) {
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
	mustPanic("Axpy", func() { Axpy(1, two, one) })
	mustPanic("Xpby", func() { Xpby(two, 1, one) })
	mustPanic("AxpyDot y", func() { AxpyDot(1, two, one, two) })
	mustPanic("AxpyDot w", func() { AxpyDot(1, two, two, one) })
}
