package floats

import (
	"math"
	"math/rand"
	"testing"
)

// TestExp32AVX2MatchesGo holds the AVX2 kernel to exp32CR bit for bit (NaN
// to NaN) on random inputs across the float32 range, the special values and
// the clamp and overflow edges, at every length 0..70 (blocks of 8 and the Go
// tail).
func TestExp32AVX2MatchesGo(t *testing.T) {
	if !useExp32AVX2 {
		t.Skip("no AVX2+FMA here")
	}
	r := rand.New(rand.NewSource(3))
	src := make([]float32, 0, 100000)
	for _, v := range []float32{0, float32(math.Copysign(0, -1)), 1, -1, 88.72283, 88.72284, 89, -103.972, -103.973, -104,
		511.9, 512, 512.1, -512, -512.1, 1e30, -1e30, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), math.Float32frombits(0xffc00000), math.SmallestNonzeroFloat32} {
		src = append(src, v)
	}
	for len(src) < cap(src) {
		src = append(src, math.Float32frombits(r.Uint32()))
	}
	want := make([]float32, len(src))
	exp32CR(want, src)
	got := make([]float32, len(src))
	exp32Fast(got, src)
	for i := range src {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) && !(got[i] != got[i] && want[i] != want[i]) {
			t.Fatalf("exp(0x%08x): AVX2 0x%08x, Go 0x%08x", math.Float32bits(src[i]), math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
	for n := 0; n <= 70; n++ {
		d := make([]float32, n)
		exp32Fast(d, src[:n])
		for i := range d {
			if math.Float32bits(d[i]) != math.Float32bits(want[i]) && !(d[i] != d[i] && want[i] != want[i]) {
				t.Fatalf("length %d [%d]: 0x%08x, want 0x%08x", n, i, math.Float32bits(d[i]), math.Float32bits(want[i]))
			}
		}
	}
}
