//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestCMulPortableForced: the path of a CPU without AVX2+FMA, forced on any host,
// held to the same bounds.
func TestCMulPortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	gatherBoundSweep(t)
	scatterBoundSweep(t)
}

// TestCMulAVX2Dispatch: on a CPU with AVX2+FMA the public functions run the
// kernels: the kernel plus the Go tail is the public result bit for bit, and the
// kernel alone is not the portable loop's result (it fuses), so the test cannot
// pass by quietly taking the portable path.
func TestCMulAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loops run, covered by TestCMulPortableForced")
	}
	rng := rand.New(rand.NewSource(65))
	differs := false
	for _, n := range twiddleSizes {
		x, c, tw := twiddleInputs(rng, n)
		x32 := to32(x)
		pub, ker := make([]float64, n), make([]float64, n)
		CMulGather(pub, x, tw)
		cmulGather64AVX2(ker, x, tw)
		cmulGatherTail(ker, x, tw, (n/2)&^3)
		pub32, ker32 := make([]float64, n), make([]float64, n)
		CMulGatherFrom32(pub32, x32, tw)
		cmulGatherFrom32AVX2(ker32, x32, tw)
		cmulGather32Tail(ker32, x32, tw, (n/2)&^3)
		for i := range pub {
			if pub[i] != ker[i] || pub32[i] != ker32[i] {
				t.Fatalf("n=%d: the public gather is not the kernel plus the tail at %d", n, i)
			}
		}
		ref := make([]float64, n)
		cmulGatherTail(ref, x, tw, 0)
		for i := range ref {
			if ref[i] != pub[i] {
				differs = true
			}
		}
		if n < 8 {
			continue
		}
		for _, f := range []float64{1, -0.3} {
			spub, sker := make([]float64, n), make([]float64, n)
			CMulScatter(spub, c, tw, f)
			cmulScatter64AVX2(sker, c, tw, f)
			spub32, sker32 := make([]float32, n), make([]float32, n)
			CMulScatterTo32(spub32, c, tw, f)
			cmulScatterTo32AVX2(sker32, c, tw, f)
			for i := range spub {
				if spub[i] != sker[i] || spub32[i] != sker32[i] {
					t.Fatalf("n=%d f=%g: the public scatter is not the kernel at %d", n, f, i)
				}
			}
		}
	}
	if !differs {
		t.Error("the kernels never differed from the portable loop: are they fusing?")
	}
}
