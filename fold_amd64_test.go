//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestMDCTFoldPortableForced: the path of a CPU without AVX2+FMA, forced on any
// host, held to the same exact results.
func TestMDCTFoldPortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	rng := rand.New(rand.NewSource(84))
	for _, n := range foldSizes {
		x := foldInputs(rng, n)
		u := foldInputs(rng, n/2)
		got, ref := make([]float64, n/2), make([]float64, n/2)
		MDCTFold(got, x)
		foldTail(ref, x, 0)
		g32, r32 := make([]float64, n/2), make([]float64, n/2)
		MDCTFoldFrom32(g32, to32(x))
		fold32Tail(r32, to32(x), 0)
		d, rd := make([]float64, n), make([]float64, n)
		MDCTUnfold(d, u)
		unfoldTail(rd, u, 0)
		d32, rd32 := make([]float32, n), make([]float32, n)
		MDCTUnfoldTo32(d32, u)
		unfold32Tail(rd32, u, 0)
		for i := range got {
			if got[i] != ref[i] || g32[i] != r32[i] {
				t.Fatalf("fold n=%d i=%d differs", n, i)
			}
		}
		for i := range d {
			if d[i] != rd[i] || d32[i] != rd32[i] {
				t.Fatalf("unfold n=%d i=%d differs", n, i)
			}
		}
	}
}

// TestMDCTFoldAVX2Dispatch: on a CPU with AVX2+FMA the public functions run the
// kernels (the kernel plus the Go tail is the public result), and the kernels agree
// with the portable loops: nothing here can pass by quietly taking the portable path
// in the first half, since the kernel is called directly.
func TestMDCTFoldAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loops run, covered by TestMDCTFoldPortableForced")
	}
	rng := rand.New(rand.NewSource(85))
	for _, n := range foldSizes {
		x := foldInputs(rng, n)
		u := foldInputs(rng, n/2)
		pub, ker, ref := make([]float64, n/2), make([]float64, n/2), make([]float64, n/2)
		MDCTFold(pub, x)
		mdctFold64AVX2(ker, x)
		foldTail(ker, x, (n/4)&^3)
		foldTail(ref, x, 0)
		x32 := to32(x)
		pub32, ker32, ref32 := make([]float64, n/2), make([]float64, n/2), make([]float64, n/2)
		MDCTFoldFrom32(pub32, x32)
		mdctFoldFrom32AVX2(ker32, x32)
		fold32Tail(ker32, x32, (n/4)&^3)
		fold32Tail(ref32, x32, 0)
		d, kd, rd := make([]float64, n), make([]float64, n), make([]float64, n)
		MDCTUnfold(d, u)
		mdctUnfold64AVX2(kd, u)
		unfoldTail(kd, u, (n/4)&^3)
		unfoldTail(rd, u, 0)
		d32, kd32, rd32 := make([]float32, n), make([]float32, n), make([]float32, n)
		MDCTUnfoldTo32(d32, u)
		mdctUnfoldTo32AVX2(kd32, u)
		unfold32Tail(kd32, u, (n/4)&^3)
		unfold32Tail(rd32, u, 0)
		for i := range pub {
			if pub[i] != ker[i] || ker[i] != ref[i] || pub32[i] != ker32[i] || ker32[i] != ref32[i] {
				t.Fatalf("fold n=%d i=%d: public %g kernel %g portable %g", n, i, pub[i], ker[i], ref[i])
			}
		}
		for i := range d {
			if d[i] != kd[i] || kd[i] != rd[i] || d32[i] != kd32[i] || kd32[i] != rd32[i] {
				t.Fatalf("unfold n=%d i=%d: public %g kernel %g portable %g", n, i, d[i], kd[i], rd[i])
			}
		}
	}
}
