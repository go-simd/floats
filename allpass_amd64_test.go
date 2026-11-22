//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestAllPass3PortableForced: the path of a CPU without AVX2+FMA, forced on any host, gives
// the same bits as the definition at every row count.
func TestAllPass3PortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestAllPass3Exact(t)
	TestAllPass3Bound(t)
}

// TestAllPass3AVX2Dispatch: on a CPU with AVX2+FMA the kernel itself equals the portable
// loop bit for bit at every row count, odd tails included.
func TestAllPass3AVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestAllPass3PortableForced")
	}
	rng := rand.New(rand.NewSource(205))
	for n := 1; n <= 70; n++ {
		f := newAPFix(rng, n, 3, n&3)
		g := f.clone()
		allPass3AVX2(n, &f.out[0], &f.in[0], &f.t.Phi[0], &f.t.Q0[0], &f.t.Q1[0], &f.t.Q2[0], &f.t.C0[0], &f.t.C1[0], &f.t.C2[0],
			&f.old[0][0], &f.cur[0][0], &f.old[1][0], &f.cur[1][0], &f.old[2][0], &f.cur[2][0])
		g.call(allPass3Ref)
		if d := f.diff(g); d != "" {
			t.Fatalf("n=%d: AVX2 kernel differs from the portable loop: %s", n, d)
		}
	}
}
