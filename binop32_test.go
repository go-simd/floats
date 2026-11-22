package floats

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// The reference of each float32 kernel is the Go expression itself, spelled out
// here (not binop32Ref): the kernels must agree with it bit for bit on every
// length, alignment and special value (NaN of either sign, signed zeros,
// infinities, subnormals), and must not write past dst.

var sentinel32 = math.Float32frombits(0x7fc0dead)

var specials32 = []float32{
	float32(math.NaN()), math.Float32frombits(0xffc00123), float32(math.Inf(1)), float32(math.Inf(-1)),
	0, float32(math.Copysign(0, -1)), math.MaxFloat32, -math.MaxFloat32,
	math.SmallestNonzeroFloat32, 1e-40, 1, -1, 0.5, 4, 9,
}

// fill32 returns n random values at element offset off of a backing slice that
// is followed by guard sentinels, so the tests see an unaligned start and any
// write past the end.
func fill32(n, off int, rng *rand.Rand, withSpecials bool) (s, buf []float32) {
	buf = make([]float32, off+n+guard)
	for i := range buf {
		buf[i] = sentinel32
	}
	s = buf[off : off+n : off+n]
	for i := range s {
		s[i] = float32(rng.NormFloat64() * math.Exp2(float64(rng.Intn(20)-10)))
		if withSpecials && rng.Intn(8) == 0 {
			s[i] = specials32[rng.Intn(len(specials32))]
		}
	}
	return s, buf
}

func checkSame32(t *testing.T, what string, got, want []float32, n, off int) {
	t.Helper()
	for i := range want {
		if !same32(got[i], want[i]) {
			t.Fatalf("%s n=%d off=%d: [%d] = %v (%#x), want %v (%#x)", what, n, off, i,
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func checkGuard32(t *testing.T, what string, buf []float32, n, off int) {
	t.Helper()
	for i := off + n; i < len(buf); i++ {
		if math.Float32bits(buf[i]) != math.Float32bits(sentinel32) {
			t.Fatalf("%s n=%d off=%d: wrote past the end at %d", what, n, off, i-off)
		}
	}
}

// forEachCase32 runs body over every length 0..257, element offsets 0..3 and
// with or without planted specials.
func forEachCase32(seed int64, body func(n, off int, specials bool, rng *rand.Rand)) {
	rng := rand.New(rand.NewSource(seed))
	for n := 0; n <= 257; n++ {
		for off := 0; off < 4; off++ {
			for _, sp := range []bool{false, true} {
				body(n, off, sp, rng)
			}
		}
	}
}

var binop32Ops = []struct {
	name string
	f    func(dst, a, b []float32)
	ref  func(dst, a, b []float32)
	expr func(x, y float32) float32
}{
	{"Add32", Add32, add32Ref, func(x, y float32) float32 { return x + y }},
	{"Sub32", Sub32, sub32Ref, func(x, y float32) float32 { return x - y }},
	{"Mul32", Mul32, mul32Ref, func(x, y float32) float32 { return x * y }},
	{"Div32", Div32, div32Ref, func(x, y float32) float32 { return x / y }},
}

func sqrt32Expr(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func TestBinop32BitIdentical(t *testing.T) {
	for _, op := range binop32Ops {
		forEachCase32(41, func(n, off int, sp bool, rng *rand.Rand) {
			a, _ := fill32(n, off, rng, sp)
			b, _ := fill32(n, (off+1)%4, rng, sp)
			dst, buf := fill32(n, (off+2)%4, rng, sp)
			want := make([]float32, n)
			for i := range want {
				want[i] = op.expr(a[i], b[i])
			}
			op.f(dst, a, b)
			checkSame32(t, op.name, dst, want, n, off)
			checkGuard32(t, op.name, buf, n, (off+2)%4)
			op.ref(dst, a, b)
			checkSame32(t, op.name+"Ref", dst, want, n, off)
		})
	}
}

func TestSqrt32BitIdentical(t *testing.T) {
	forEachCase32(42, func(n, off int, sp bool, rng *rand.Rand) {
		a, _ := fill32(n, off, rng, sp)
		dst, buf := fill32(n, (off+2)%4, rng, sp)
		want := make([]float32, n)
		for i := range want {
			want[i] = sqrt32Expr(a[i])
		}
		Sqrt32(dst, a)
		checkSame32(t, "Sqrt32", dst, want, n, off)
		checkGuard32(t, "Sqrt32", buf, n, (off+2)%4)
		sqrt32Ref(dst, a)
		checkSame32(t, "sqrt32Ref", dst, want, n, off)
	})
}

// dst may be the very slice of an input.
func TestBinop32Aliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	n := 70
	a, _ := fill32(n, 0, rng, true)
	b, _ := fill32(n, 1, rng, true)
	want := make([]float32, n)
	for _, op := range binop32Ops {
		for i := range want {
			want[i] = op.expr(a[i], b[i])
		}
		x := append([]float32(nil), a...)
		op.f(x, x, b)
		checkSame32(t, op.name+" alias a", x, want, n, 0)
		y := append([]float32(nil), b...)
		op.f(y, a, y)
		checkSame32(t, op.name+" alias b", y, want, n, 0)
		z := append([]float32(nil), a...)
		op.f(z, z, z)
		for i := range want {
			want[i] = op.expr(a[i], a[i])
		}
		checkSame32(t, op.name+" alias both", z, want, n, 0)
	}
	x := append([]float32(nil), a...)
	Sqrt32(x, x)
	for i := range x {
		want[i] = sqrt32Expr(a[i])
	}
	checkSame32(t, "Sqrt32 alias", x, want, n, 0)
}

// Only the first len(a) elements are touched, and a short operand panics.
func TestBinop32Lengths(t *testing.T) {
	a, b := []float32{1, 2, 3}, []float32{4, 5, 6, 7, 8}
	dst := []float32{9, 9, 9, 9, 9}
	for _, c := range []struct {
		name string
		f    func(dst, a, b []float32)
		want [3]float32
	}{
		{"Add32", Add32, [3]float32{5, 7, 9}},
		{"Sub32", Sub32, [3]float32{-3, -3, -3}},
		{"Mul32", Mul32, [3]float32{4, 10, 18}},
		{"Div32", Div32, [3]float32{0.25, 0.4, 0.5}},
	} {
		c.f(dst, a, b)
		if dst[0] != c.want[0] || dst[1] != c.want[1] || dst[2] != c.want[2] || dst[3] != 9 || dst[4] != 9 {
			t.Fatalf("%s touched or computed wrongly: %v", c.name, dst)
		}
	}
	Sqrt32(dst, []float32{4, 9})
	if dst[0] != 2 || dst[1] != 3 || dst[2] != 0.5 {
		t.Fatalf("Sqrt32: %v", dst)
	}
	short := []float32{1, 2}
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		f()
	}
	for _, op := range binop32Ops {
		mustPanic(op.name+" b", func() { op.f(dst, a, short) })
		mustPanic(op.name+" dst", func() { op.f(short, a, b) })
	}
	mustPanic("Sqrt32 dst", func() { Sqrt32(short, a) })
}

// Sqrt32 on exact squares and the special values (the exhaustive proof over all
// 2^32 inputs is sqrt32_exhaustive_test.go).
func TestSqrt32Specials(t *testing.T) {
	in := []float32{0, float32(math.Copysign(0, -1)), 1, 4, 16, 2, 3, 0.25, math.MaxFloat32,
		math.SmallestNonzeroFloat32, 1e-40, float32(math.Inf(1)), float32(math.Inf(-1)), -1, -1e-45,
		float32(math.NaN()), math.Float32frombits(0xffc00001)}
	got := make([]float32, len(in))
	Sqrt32(got, in)
	for i, x := range in {
		if w := sqrt32Expr(x); !same32(got[i], w) {
			t.Errorf("Sqrt32(%v) = %v (%#x), want %v (%#x)", x, got[i], math.Float32bits(got[i]), w, math.Float32bits(w))
		}
	}
	if g := got[1]; math.Float32bits(g) != 0x80000000 {
		t.Errorf("Sqrt32(-0) = %#x, want -0", math.Float32bits(g))
	}
	if g := got[len(in)-4]; g == g { // sqrt(-1)
		t.Errorf("Sqrt32(-1) = %v, want NaN", g)
	}
}

// rawF32 decodes the fuzz corpus into float32 bit patterns, NaNs and infinities
// included.
func rawF32(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

// FuzzBinop32 checks Add32, Sub32, Mul32, Div32 and Sqrt32 against their
// definitions bit for bit (NaN payloads of arithmetic results aside) on
// arbitrary bit patterns at four start offsets.
func FuzzBinop32(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add(uint8(1), make([]byte, 4*45))
	f.Add(uint8(3), []byte{0, 0, 0xc0, 0x7f, 0, 0, 0x80, 0xff, 0, 0, 0x80, 0x7f, 1, 0, 0, 0, 0, 0, 0, 0x80})
	f.Fuzz(func(t *testing.T, offset uint8, data []byte) {
		v := rawF32(data)
		off := int(offset % 4)
		if off > len(v) {
			off = len(v)
		}
		v = v[off:]
		n := len(v) / 2
		a, b := v[:n], v[n:2*n]
		got, want := make([]float32, n), make([]float32, n)
		for _, op := range binop32Ops {
			for i := range want {
				want[i] = op.expr(a[i], b[i])
			}
			op.f(got, a, b)
			checkSame32(t, "Fuzz"+op.name, got, want, n, off)
		}
		for i := range want {
			want[i] = sqrt32Expr(a[i])
		}
		Sqrt32(got, a)
		checkSame32(t, "FuzzSqrt32", got, want, n, off)
	})
}
