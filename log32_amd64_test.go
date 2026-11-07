package floats

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestLogCRConstsLayout holds logCRConsts to the offsets log32CRKernel uses.
func TestLogCRConstsLayout(t *testing.T) {
	var k logCRConsts
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"one", unsafe.Offsetof(k.one), 0}, {"ln2", unsafe.Offsetof(k.ln2), 32},
		{"b0", unsafe.Offsetof(k.b0), 64}, {"b1", unsafe.Offsetof(k.b1), 96},
		{"b2", unsafe.Offsetof(k.b2), 128}, {"w", unsafe.Offsetof(k.w), 160},
		{"sub", unsafe.Offsetof(k.sub), 192}, {"maxv", unsafe.Offsetof(k.maxv), 224},
		{"mmask", unsafe.Offsetof(k.mmask), 256}, {"jadd", unsafe.Offsetof(k.jadd), 288},
		{"bias", unsafe.Offsetof(k.bias), 320}, {"oneBits", unsafe.Offsetof(k.oneBits), 352},
		{"expOr", unsafe.Offsetof(k.expOr), 384}, {"tr", unsafe.Offsetof(k.tr), 416},
		{"tl", unsafe.Offsetof(k.tl), 936},
	} {
		if c.got != c.want {
			t.Errorf("%s at %d, the kernel reads %d", c.name, c.got, c.want)
		}
	}
}

// TestLog32AVX2MatchesGo holds log32Fast (AVX2 blocks, Go for the blocks it
// hands back and the tail) to logCR bit for bit (NaN to NaN): random bit
// patterns (a third of them negative, NaN or subnormal, so most blocks bail),
// random positive normals (few bail), 1.0 runs, the special values, and every
// length 0..70.
func TestLog32AVX2MatchesGo(t *testing.T) {
	if !useLog32AVX2 {
		t.Skip("no AVX2 here")
	}
	r := rand.New(rand.NewSource(5))
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
	got := make([]float32, len(src))
	log32Fast(got, src)
	same := func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b) }
	for i := range src {
		if !same(got[i], want[i]) {
			t.Fatalf("log(0x%08x): AVX2 0x%08x, Go 0x%08x", math.Float32bits(src[i]), math.Float32bits(got[i]), math.Float32bits(want[i]))
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
}

// TestLog32AVX2InPlace runs the kernel with dst == src: a block it hands
// back must still hold its inputs.
func TestLog32AVX2InPlace(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	src := make([]float32, 4099)
	for i := range src {
		src[i] = math.Float32frombits(r.Uint32() &^ (1 << 31)) // NaN, Inf and subnormals among them
	}
	want := make([]float32, len(src))
	logCRSlice(want, src)
	log32Fast(src, src)
	for i := range src {
		if math.Float32bits(src[i]) != math.Float32bits(want[i]) && !(src[i] != src[i] && want[i] != want[i]) {
			t.Fatalf("[%d]: 0x%08x, want 0x%08x", i, math.Float32bits(src[i]), math.Float32bits(want[i]))
		}
	}
}
