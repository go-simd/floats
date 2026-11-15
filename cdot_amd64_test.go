//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestCDotCPortableForced: the path of a CPU without AVX2+FMA, forced on any host, is
// within the same bound.
func TestCDotCPortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestCDotCBound(t)
}

// TestCDotCAVX2Dispatch: on a CPU with AVX2+FMA the public function runs the kernel
// (kernel plus the Go tail is the public result bit for bit), and it is within the bound.
// It fails, not skips, on the CI amd64 lanes (GITHUB_ACTIONS set), which have AVX2.
func TestCDotCAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		if onCI() {
			t.Fatal("no AVX2+FMA on a CI amd64 runner: the kernel would go untested")
		}
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestCDotCPortableForced")
	}
	rng := rand.New(rand.NewSource(304))
	for _, n := range cdotLengths() {
		a, b := cdotInputs(rng, 2*n), cdotInputs(rng, 2*n)
		pr, pi := CDotC(a, b)
		var out [8]float64
		cdotcAVX2(a, b, out[:])
		kr, ki := (out[0]+out[1])+(out[2]+out[3]), (out[4]-out[5])+(out[6]-out[7])
		tr, ti := cdotcTail(a, b, len(a)&^7)
		if pr != kr+tr || pi != ki+ti {
			t.Fatalf("n=%d: the public result is not the kernel plus the tail", n)
		}
		cdotCheck(t, "AVX2", n, pr, pi, a, b)
	}
}
