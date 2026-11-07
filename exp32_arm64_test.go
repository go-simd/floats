package floats

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestExpCRNEONConstsLayout holds expCRNEONConsts to the offsets
// exp32CRKernel loads.
func TestExpCRNEONConstsLayout(t *testing.T) {
	var k expCRNEONConsts
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"shift", unsafe.Offsetof(k.shift), 0}, {"ln2Lo", unsafe.Offsetof(k.ln2Lo), 48},
		{"c0", unsafe.Offsetof(k.c0), 64}, {"c3", unsafe.Offsetof(k.c3), 112},
		{"idxMask", unsafe.Offsetof(k.idxMask), 128}, {"expMask", unsafe.Offsetof(k.expMask), 144},
		{"lim", unsafe.Offsetof(k.lim), 160}, {"table", unsafe.Offsetof(k.table), 176},
	} {
		if c.got != c.want {
			t.Errorf("%s at %d, the kernel reads %d", c.name, c.got, c.want)
		}
	}
}

// TestExp32NEONMatchesGo holds exp32Fast (NEON blocks, Go for the blocks it
// hands back and the tail) to exp32CR bit for bit (NaN to NaN): random bit
// patterns (most blocks hand back), random inputs in [-110, 110] (none
// does), the special values and the ±512 edge, every length 0..70, and in
// place.
func TestExp32NEONMatchesGo(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	src := make([]float32, 0, 300000)
	for _, v := range []float32{0, float32(math.Copysign(0, -1)), 1, -1, 88.72283, 88.72284, 89, -103.972, -103.973, -104,
		511.9, 512, 512.1, -512, -512.1, 1e30, -1e30, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), math.Float32frombits(0xffc00000), math.SmallestNonzeroFloat32} {
		src = append(src, v)
	}
	for len(src) < 100000 {
		src = append(src, math.Float32frombits(r.Uint32()))
	}
	for len(src) < cap(src) {
		src = append(src, float32(r.Float64()*220-110))
	}
	want := make([]float32, len(src))
	exp32CR(want, src)
	same := func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b) }
	got := make([]float32, len(src))
	exp32Fast(got, src)
	for i := range src {
		if !same(got[i], want[i]) {
			t.Fatalf("exp(0x%08x): NEON 0x%08x, Go 0x%08x", math.Float32bits(src[i]), math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
	for n := 0; n <= 70; n++ {
		d := make([]float32, n)
		exp32Fast(d, src[:n])
		for i := range d {
			if !same(d[i], want[i]) {
				t.Fatalf("length %d [%d]: 0x%08x, want 0x%08x", n, i, math.Float32bits(d[i]), math.Float32bits(want[i]))
			}
		}
	}
	in := append([]float32(nil), src[:4099]...)
	exp32Fast(in, in)
	for i := range in {
		if !same(in[i], want[i]) {
			t.Fatalf("in place [%d]: 0x%08x, want 0x%08x", i, math.Float32bits(in[i]), math.Float32bits(want[i]))
		}
	}
}
