package floats

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestLogCRNEONConstsLayout holds logCRNEONConsts to the offsets
// log32CRKernel loads.
func TestLogCRNEONConstsLayout(t *testing.T) {
	var k logCRNEONConsts
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"one", unsafe.Offsetof(k.one), 0}, {"w", unsafe.Offsetof(k.w), 80},
		{"mmask", unsafe.Offsetof(k.mmask), 96}, {"oneBits", unsafe.Offsetof(k.oneBits), 176},
		{"expOr", unsafe.Offsetof(k.expOr), 192}, {"tr", unsafe.Offsetof(k.tr), 208},
		{"tl", unsafe.Offsetof(k.tl), 728},
	} {
		if c.got != c.want {
			t.Errorf("%s at %d, the kernel reads %d", c.name, c.got, c.want)
		}
	}
}

// TestLog32NEONMatchesGo holds log32Fast (NEON blocks, Go for the blocks it
// hands back and the tail) to logCR bit for bit (NaN to NaN): random bit
// patterns (most blocks hand back), random positive normals (few do), 1.0
// runs, the special values, every length 0..70, and in place.
func TestLog32NEONMatchesGo(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	src := make([]float32, 0, 300000)
	for _, v := range []float32{1, 1, 1, 1, 1, 1, 1, 1, 0, float32(math.Copysign(0, -1)), -1,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), math.SmallestNonzeroFloat32,
		0x1p-126, math.MaxFloat32, 0.99999994, 1.0000001, 2, 0.5} {
		src = append(src, v)
	}
	for len(src) < 100000 {
		src = append(src, math.Float32frombits(r.Uint32()))
	}
	for len(src) < cap(src) {
		src = append(src, math.Float32frombits(0x00800000+uint32(r.Int63n(0x7f000000))))
	}
	want := make([]float32, len(src))
	logCRSlice(want, src)
	same := func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b) }
	got := make([]float32, len(src))
	log32Fast(got, src)
	for i := range src {
		if !same(got[i], want[i]) {
			t.Fatalf("log(0x%08x): NEON 0x%08x, Go 0x%08x", math.Float32bits(src[i]), math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
	for n := 0; n <= 70; n++ {
		d := make([]float32, n)
		log32Fast(d, src[:n])
		for i := range d {
			if !same(d[i], want[i]) {
				t.Fatalf("length %d [%d]: 0x%08x, want 0x%08x", n, i, math.Float32bits(d[i]), math.Float32bits(want[i]))
			}
		}
	}
	in := append([]float32(nil), src[:4099]...)
	log32Fast(in, in)
	for i := range in {
		if !same(in[i], want[i]) {
			t.Fatalf("in place [%d]: 0x%08x, want 0x%08x", i, math.Float32bits(in[i]), math.Float32bits(want[i]))
		}
	}
}
