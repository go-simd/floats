package floats

import (
	"math"
	"testing"
)

// NumPy 2.5.4: max of any mix containing a +0 is +0 (of only -0s, -0); min of
// any mix containing a -0 is -0 (of only +0s, +0). NaN anywhere propagates.
// The zero is placed at every lane position of every length around the vector
// widths so a kernel cannot hide a lane-order dependence.

func TestMaxMinSignedZeros(t *testing.T) {
	fills := []float64{-7, -1, 3, 0, negZero}
	for n := 1; n <= 70; n++ {
		for _, fill := range fills {
			for pos := 0; pos < n; pos++ {
				for _, z := range []float64{0, negZero} {
					a := make([]float64, n)
					for i := range a {
						a[i] = fill
					}
					a[pos] = z
					wmax, wmin := refMax(a), refMin(a)
					if g := Max(a); math.Float64bits(g) != math.Float64bits(wmax) {
						t.Fatalf("Max n=%d fill=%v a[%d]=%v: got %v want %v", n, fill, pos, z, g, wmax)
					}
					if g := Min(a); math.Float64bits(g) != math.Float64bits(wmin) {
						t.Fatalf("Min n=%d fill=%v a[%d]=%v: got %v want %v", n, fill, pos, z, g, wmin)
					}
					f := make([]float32, n)
					for i := range a {
						f[i] = float32(a[i])
					}
					if g := Float32Max(f); math.Float32bits(g) != math.Float32bits(float32(wmax)) {
						t.Fatalf("Float32Max n=%d fill=%v a[%d]=%v: got %v want %v", n, fill, pos, z, g, wmax)
					}
					if g := Float32Min(f); math.Float32bits(g) != math.Float32bits(float32(wmin)) {
						t.Fatalf("Float32Min n=%d fill=%v a[%d]=%v: got %v want %v", n, fill, pos, z, g, wmin)
					}
				}
			}
		}
	}
}

func TestMaxMinNumPyZeroTable(t *testing.T) {
	p, m := 0.0, negZero
	cases := []struct {
		a        []float64
		max, min float64
	}{
		{[]float64{p, m}, p, m},
		{[]float64{m, p}, p, m},
		{[]float64{m, m}, m, m},
		{[]float64{p, p}, p, p},
		{[]float64{-1, m, -3}, m, -3},
		{[]float64{1, p, 3}, 3, p},
		{[]float64{-1, -7, m, -3, p}, p, -7},
		{[]float64{-1, -7, m, -3}, m, -7},
	}
	for _, c := range cases {
		if g := Max(c.a); math.Float64bits(g) != math.Float64bits(c.max) {
			t.Errorf("Max(%v)=%v want %v", c.a, g, c.max)
		}
		if g := Min(c.a); math.Float64bits(g) != math.Float64bits(c.min) {
			t.Errorf("Min(%v)=%v want %v", c.a, g, c.min)
		}
	}
}

func TestMaxMinNaNEveryPosition(t *testing.T) {
	for n := 1; n <= 70; n++ {
		for pos := 0; pos < n; pos++ {
			for _, z := range []float64{0, negZero, 5} {
				a := make([]float64, n)
				for i := range a {
					a[i] = z
				}
				a[pos] = math.NaN()
				if !math.IsNaN(Max(a)) || !math.IsNaN(Min(a)) {
					t.Fatalf("n=%d NaN@%d fill=%v: not NaN", n, pos, z)
				}
				f := make([]float32, n)
				for i := range f {
					f[i] = float32(z)
				}
				f[pos] = float32(math.NaN())
				if !isNaN32(Float32Max(f)) || !isNaN32(Float32Min(f)) {
					t.Fatalf("f32 n=%d NaN@%d fill=%v: not NaN", n, pos, z)
				}
			}
		}
	}
}

// refMax and refMin order the zeros by their sign bit, independently of the
// builtins the implementation uses.
func refMax(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		switch {
		case math.IsNaN(v):
			return v
		case v > m, v == 0 && m == 0 && !math.Signbit(v):
			m = v
		}
	}
	return m
}

func refMin(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		switch {
		case math.IsNaN(v):
			return v
		case v < m, v == 0 && m == 0 && math.Signbit(v):
			m = v
		}
	}
	return m
}
