//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestSquaredMagnitudePortableForced: the path of a CPU without AVX2+FMA, forced on
// any host, held to the same bound (the lengths and the oracle of the main test).
func TestSquaredMagnitudePortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestSquaredMagnitudeBound(t)
}

// TestSquaredMagnitudeAVX2Dispatch: on a CPU with AVX2+FMA the public function runs
// the kernel (kernel plus the Go tail is the public result bit for bit) and the
// kernel fuses, so it differs from the portable loop somewhere.
func TestSquaredMagnitudeAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestSquaredMagnitudePortableForced")
	}
	rng := rand.New(rand.NewSource(94))
	differs := false
	for _, n := range sqmagLengths() {
		c := sqmagInputs(rng, n)
		pub, ker, ref := make([]float64, n), make([]float64, n), make([]float64, n)
		SquaredMagnitude(pub, c, 0.37)
		sqmagAVX2(ker, c, 0.37)
		sqmagTail(ker, c, 0.37, n&^3)
		sqmagTail(ref, c, 0.37, 0)
		for i := range pub {
			if pub[i] != ker[i] {
				t.Fatalf("n=%d bin %d: the public result is not the kernel plus the tail", n, i)
			}
			if pub[i] != ref[i] {
				differs = true
			}
		}
	}
	if !differs {
		t.Error("the kernel never differed from the portable loop: is it fusing?")
	}
}
