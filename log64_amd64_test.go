package floats

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestLogConstsLayout holds logConsts to the offsets logKernel uses.
func TestLogConstsLayout(t *testing.T) {
	var k logConsts
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"lo", unsafe.Offsetof(k.lo), 0}, {"n1hi", unsafe.Offsetof(k.n1hi), 96},
		{"off", unsafe.Offsetof(k.off), 128}, {"topMask", unsafe.Offsetof(k.topMask), 192},
		{"magic", unsafe.Offsetof(k.magic), 224}, {"magicSub", unsafe.Offsetof(k.magicSub), 256},
		{"x2048", unsafe.Offsetof(k.x2048), 288}, {"negOne", unsafe.Offsetof(k.negOne), 320},
		{"ln2lo", unsafe.Offsetof(k.ln2lo), 384}, {"a0", unsafe.Offsetof(k.a0), 416},
		{"a4", unsafe.Offsetof(k.a4), 544},
	} {
		if c.got != c.want {
			t.Errorf("%s at %d, the kernel reads %d", c.name, c.got, c.want)
		}
	}
	if unsafe.Sizeof(logEntry{}) != 16 {
		t.Errorf("logEntry is %d bytes, the kernel gathers at 16i", unsafe.Sizeof(logEntry{}))
	}
}

// TestLogAVX2MatchesGo holds logFast (AVX2 blocks, Go for the blocks it hands
// back and the tail) to logScalar bit for bit (NaN to NaN): random bit
// patterns (most blocks hand back), positive normals across the whole
// exponent range (few do, and every k from -1022 to 1023 and every table
// index occurs), values near 1, the special values, every length 0..70, and
// in place.
func TestLogAVX2MatchesGo(t *testing.T) {
	if !useLogAVX2 {
		t.Skip("no AVX2+FMA here")
	}
	r := rand.New(rand.NewSource(9))
	src := make([]float64, 0, 300000)
	for _, v := range []float64{1, 1, 1, 1, 0, math.Copysign(0, -1), -1, math.Inf(1), math.Inf(-1), math.NaN(),
		5e-324, 0x1p-1022, math.MaxFloat64, 1 - 0x1p-4, 1 + 0x1.09p-4, math.Nextafter(1-0x1p-4, 0), 0.25, 24.25} {
		src = append(src, v)
	}
	for len(src) < 100000 {
		src = append(src, math.Float64frombits(r.Uint64()))
	}
	for len(src) < 250000 {
		src = append(src, math.Float64frombits(0x0010<<48+uint64(r.Int63n(0x7fe0<<48))))
	}
	for len(src) < cap(src) {
		src = append(src, 0.9+0.2*r.Float64())
	}
	want := make([]float64, len(src))
	logScalar(want, src)
	same := func(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) || (a != a && b != b) }
	got := make([]float64, len(src))
	logFast(got, src)
	for i := range src {
		if !same(got[i], want[i]) {
			t.Fatalf("log(0x%016x): AVX2 0x%016x, Go 0x%016x", math.Float64bits(src[i]), math.Float64bits(got[i]), math.Float64bits(want[i]))
		}
	}
	for n := 0; n <= 70; n++ {
		d := make([]float64, n)
		logFast(d, src[:n])
		for i := range d {
			if !same(d[i], want[i]) {
				t.Fatalf("length %d [%d]: 0x%016x, want 0x%016x", n, i, math.Float64bits(d[i]), math.Float64bits(want[i]))
			}
		}
	}
	in := append([]float64(nil), src[100000:104099]...)
	logFast(in, in)
	for i := range in {
		if !same(in[i], want[100000+i]) {
			t.Fatalf("in place [%d]: 0x%016x, want 0x%016x", i, math.Float64bits(in[i]), math.Float64bits(want[100000+i]))
		}
	}
}
