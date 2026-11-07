package floats

import (
	"math"
	"math/rand"
	"testing"
)

// ulps32 is the distance between two float32s in units in the last place
// (0 for equal values, including two NaNs).
func ulps32(a, b float32) int64 {
	if a == b || (a != a && b != b) {
		return 0
	}
	ia, ib := int64(math.Float32bits(a)), int64(math.Float32bits(b))
	if ia < 0 {
		ia = math.MinInt32 - ia
	}
	if ib < 0 {
		ib = math.MinInt32 - ib
	}
	if d := ia - ib; d < 0 {
		return -d
	} else {
		return d
	}
}

// TestVia64MatchesMath holds Exp32 and Log32 to float32(math.Exp/Log(x)):
// both round a float64 result once, so they may differ by one float32 ULP
// where math's float64 result (0.88 ULP) and ours (0.504) fall on opposite
// sides of a rounding midpoint, and must agree on at least 99.9% of inputs.
// Every length 0..600 crosses the 256-element chunk boundaries.
func TestVia64MatchesMath(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, f := range []struct {
		name string
		k    func(dst, src []float32)
		ref  func(float64) float64
		gen  func() float32
	}{
		{"Exp", Float32Exp, math.Exp, func() float32 { return float32(r.Float64()*200 - 110) }},
		{"Log", Float32Log, math.Log, func() float32 {
			return float32(math.Exp(r.Float64()*170 - 85))
		}},
	} {
		const n = 100000
		src, dst := make([]float32, n), make([]float32, n)
		for i := range src {
			src[i] = f.gen()
		}
		f.k(dst, src)
		var off int
		for i, x := range src {
			want := float32(f.ref(float64(x)))
			switch d := ulps32(dst[i], want); {
			case d > 1:
				t.Fatalf("%s(%v) = %v, math gives %v (%d ULP)", f.name, x, dst[i], want, d)
			case d == 1:
				off++
			}
		}
		if off > n/1000 {
			t.Errorf("%s: %d of %d differ from math by one ULP", f.name, off, n)
		}
		for m := 0; m <= 600; m++ {
			d := make([]float32, m)
			f.k(d, src[:m])
			for i := range d {
				if d[i] != dst[i] {
					t.Fatalf("%s length %d [%d]: %v != %v", f.name, m, i, d[i], dst[i])
				}
			}
		}
	}
}

// TestVia64Specials checks the edges that come from rounding the float64
// result: overflow to +Inf, underflow through the subnormals to 0, and the
// IEEE special values.
func TestVia64Specials(t *testing.T) {
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	negZero := float32(math.Copysign(0, -1))
	for _, c := range []struct {
		name    string
		k       func(dst, src []float32)
		in, out float32
	}{
		{"Exp", Float32Exp, 0, 1},
		{"Exp", Float32Exp, negZero, 1},
		{"Exp", Float32Exp, 89, inf}, // finite in float64, overflows float32
		{"Exp", Float32Exp, -inf, 0},
		{"Exp", Float32Exp, inf, inf},
		{"Exp", Float32Exp, -104, 0}, // below the smallest float32 subnormal
		{"Exp", Float32Exp, -100, float32(math.Exp(-100))},
		{"Log", Float32Log, 1, 0},
		{"Log", Float32Log, 0, -inf},
		{"Log", Float32Log, negZero, -inf},
		{"Log", Float32Log, inf, inf},
		{"Log", Float32Log, math.SmallestNonzeroFloat32, float32(math.Log(math.SmallestNonzeroFloat32))},
	} {
		d := []float32{0}
		c.k(d, []float32{c.in})
		if d[0] != c.out || math.Signbit(float64(d[0])) != math.Signbit(float64(c.out)) {
			t.Errorf("%s(%v) = %v, want %v", c.name, c.in, d[0], c.out)
		}
	}
	for _, in := range []float32{nan, -1, -inf} {
		d := []float32{0}
		Float32Log(d, []float32{in})
		if d[0] == d[0] {
			t.Errorf("Log(%v) = %v, want NaN", in, d[0])
		}
	}
	d := []float32{0}
	Float32Exp(d, []float32{nan})
	if d[0] == d[0] {
		t.Errorf("Exp(NaN) = %v", d[0])
	}
}

// TestVia64InPlace: dst may be src, a long dst is left untouched past len(src),
// and a short dst panics.
func TestVia64InPlace(t *testing.T) {
	src := make([]float32, 3*1024+17)
	for i := range src {
		src[i] = float32(i%97) * 0.25
	}
	want := make([]float32, len(src))
	Float32Exp(want, src)
	got := append([]float32(nil), src...)
	Float32Exp(got, got)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]: in place %v, out of place %v", i, got[i], want[i])
		}
	}
	long := []float32{7, 7, 7}
	Float32Log(long, []float32{1, 1})
	Float32Exp(long, []float32{0, 0})
	if long[2] != 7 {
		t.Fatalf("wrote past len(src): %v", long)
	}
	for name, f := range map[string]func(dst, src []float32){"Float32Exp": Float32Exp, "Float32Log": Float32Log} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s with a short dst did not panic", name)
				}
			}()
			f(make([]float32, 1), []float32{1, 1})
		}()
	}
}
