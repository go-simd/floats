package floats

import (
	"encoding/binary"
	"math"
	"testing"
)

// The float32 Max/Min contract (numpy.max/numpy.min): NaN if any element is NaN,
// the extremum otherwise, with -0 ordered below +0. The kernels are checked
// against an oracle that orders the zeros by their sign bit, independently of the
// builtins the implementation uses, at every length around the vector widths and
// with the interesting element at every position (every lane, every tail slot).

var nan32 = float32(math.NaN())

// refMax32 and refMin32: one sequential pass.
func refMax32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		switch {
		case v != v:
			return v
		case m != m:
			return m
		case v > m, v == 0 && m == 0 && !math.Signbit(float64(v)):
			m = v
		}
	}
	return m
}

func refMin32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		switch {
		case v != v:
			return v
		case m != m:
			return m
		case v < m, v == 0 && m == 0 && math.Signbit(float64(v)):
			m = v
		}
	}
	return m
}

// same reports bit equality, with every NaN equal to every other.
func same32(g, w float32) bool {
	if g != g || w != w {
		return g != g && w != w
	}
	return math.Float32bits(g) == math.Float32bits(w)
}

func checkExtrema32(t *testing.T, what string, a []float32) {
	t.Helper()
	if g, w := Float32Max(a), refMax32(a); !same32(g, w) {
		t.Fatalf("%s: Float32Max n=%d: got %v want %v", what, len(a), g, w)
	}
	if g, w := Float32Min(a), refMin32(a); !same32(g, w) {
		t.Fatalf("%s: Float32Min n=%d: got %v want %v", what, len(a), g, w)
	}
	if g, w := maxRef32(a), refMax32(a); !same32(g, w) {
		t.Fatalf("%s: maxRef32 n=%d: got %v want %v", what, len(a), g, w)
	}
	if g, w := minRef32(a), refMin32(a); !same32(g, w) {
		t.Fatalf("%s: minRef32 n=%d: got %v want %v", what, len(a), g, w)
	}
}

// extremaMaxLen is more than two of the widest vector block (16 floats) plus tails.
const extremaMaxLen = 100

func TestFloat32ExtremaNaNEveryPosition(t *testing.T) {
	for n := 1; n <= extremaMaxLen; n++ {
		for pos := 0; pos < n; pos++ {
			for _, fill := range []float32{0, negZero32, 5, -5} {
				a := make([]float32, n)
				for i := range a {
					a[i] = fill
				}
				a[pos] = nan32
				if !isNaN32(Float32Max(a)) || !isNaN32(Float32Min(a)) {
					t.Fatalf("n=%d NaN@%d fill=%v: not NaN", n, pos, fill)
				}
			}
		}
	}
}

func TestFloat32ExtremaSignedZeros(t *testing.T) {
	fills := []float32{-7, -1, 3, 0, negZero32}
	for n := 1; n <= extremaMaxLen; n++ {
		for _, fill := range fills {
			for pos := 0; pos < n; pos++ {
				for _, z := range []float32{0, negZero32} {
					a := make([]float32, n)
					for i := range a {
						a[i] = fill
					}
					a[pos] = z
					checkExtrema32(t, "zero", a)
				}
			}
		}
	}
}

// negZero32 is -0 as a float32 constant expression cannot be.
var negZero32 = math.Float32frombits(0x80000000)

func TestFloat32ExtremaNumPyZeroTable(t *testing.T) {
	p, m := float32(0), negZero32
	cases := []struct {
		a        []float32
		max, min float32
	}{
		{[]float32{p, m}, p, m},
		{[]float32{m, p}, p, m},
		{[]float32{m, m}, m, m},
		{[]float32{p, p}, p, p},
		{[]float32{-1, m, -3}, m, -3},
		{[]float32{1, p, 3}, 3, p},
		{[]float32{-1, -7, m, -3, p}, p, -7},
		{[]float32{-1, -7, m, -3}, m, -7},
	}
	for _, c := range cases {
		if g := Float32Max(c.a); math.Float32bits(g) != math.Float32bits(c.max) {
			t.Errorf("Float32Max(%v)=%v want %v", c.a, g, c.max)
		}
		if g := Float32Min(c.a); math.Float32bits(g) != math.Float32bits(c.min) {
			t.Errorf("Float32Min(%v)=%v want %v", c.a, g, c.min)
		}
	}
}

// TestFloat32ExtremaRandom: slices of ±Inf, ±0, repeated and denormal values, every
// length from 1 to 200 (0 panics, below).
func TestFloat32ExtremaRandom(t *testing.T) {
	inf := float32(math.Inf(1))
	vals := []float32{inf, -inf, 0, negZero32, 1, -1, 2.5, -2.5, 3e38, -3e38, 1e-45, -1e-45}
	seed := uint64(88172645463325252)
	next := func() int {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		return int(seed % uint64(len(vals)))
	}
	for n := 1; n <= 200; n++ {
		for rep := 0; rep < 20; rep++ {
			a := make([]float32, n)
			for i := range a {
				a[i] = vals[next()]
			}
			checkExtrema32(t, "random", a)
		}
	}
}

// TestFloat32ExtremaPlantedExtremum puts the one largest (smallest) element at
// every position of every length, among values that are all distinct.
func TestFloat32ExtremaPlantedExtremum(t *testing.T) {
	for n := 1; n <= extremaMaxLen; n++ {
		for pos := 0; pos < n; pos++ {
			a := make([]float32, n)
			for i := range a {
				a[i] = float32(i%13) - 6
			}
			a[pos] = 1000
			if g := Float32Max(a); g != 1000 {
				t.Fatalf("n=%d max@%d: got %v", n, pos, g)
			}
			a[pos] = -1000
			if g := Float32Min(a); g != -1000 {
				t.Fatalf("n=%d min@%d: got %v", n, pos, g)
			}
		}
	}
}

func TestFloat32ExtremaEmptyPanics(t *testing.T) {
	for name, f := range map[string]func(){
		"Float32Max": func() { Float32Max(nil) },
		"Float32Min": func() { Float32Min([]float32{}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s of an empty slice did not panic", name)
				}
			}()
			f()
		}()
	}
}

// f32sFromBytes decodes raw bits, NaN payloads and all.
func f32sFromBytes(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// FuzzExtrema32 drives Float32Max/Float32Min with arbitrary bit patterns (NaNs of
// every payload, infinities, zeros, denormals) and checks them against the
// sequential oracle.
func FuzzExtrema32(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0x80})
	f.Add(make([]byte, 4*40))
	nan := make([]byte, 4*40)
	nan[4*37+2], nan[4*37+3] = 0xc0, 0x7f
	f.Add(nan)
	f.Fuzz(func(t *testing.T, data []byte) {
		a := f32sFromBytes(data)
		if len(a) == 0 {
			return
		}
		checkExtrema32(t, "fuzz", a)
	})
}
