//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestConjReversePortableForced: the path of a CPU without AVX2, forced on any host,
// held to the same exact result.
func TestConjReversePortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestConjReverseExact(t)
}

// TestConjReverseAVX2Dispatch: on a CPU with AVX2 the public function runs the kernel
// (kernel plus the Go tail is the public result bit for bit).
func TestConjReverseAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestConjReversePortableForced")
	}
	rng := rand.New(rand.NewSource(104))
	for _, n := range conjRevLengths() {
		src := conjRevInputs(rng, n)
		pub, ker := make([]float64, 2*n), make([]float64, 2*n)
		ConjReverse(pub, src)
		if n > 0 {
			conjRevAVX2(ker, src)
			conjRevTail(ker, src, n&^3)
		}
		if i := conjBitDiff(pub, ker); i >= 0 {
			t.Fatalf("n=%d element %d: the public result is not the kernel plus the tail", n, i)
		}
	}
}
