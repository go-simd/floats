//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestMulAddRows32PortableForced: the path of a CPU without AVX2+FMA, forced on any host,
// holds the bound at every shape.
func TestMulAddRows32PortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestMulAddRows32AgainstExact(t)
}

// TestMulAddRows32AVX2Dispatch: on a CPU with AVX2+FMA the kernel agrees with the portable
// loop within twice the bound at every shape (they differ only by a double rounding).
func TestMulAddRows32AVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestMulAddRows32PortableForced")
	}
	rng := rand.New(rand.NewSource(305))
	for _, s := range mulRowsShapes() {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := mulRows32Inputs(rng, (rows-1)*lda+n)
		b := mulRows32Inputs(rng, (rows-1)*ldb+n)
		d0 := mulRows32Inputs(rng, n)
		ker, ref := append([]float32(nil), d0...), append([]float32(nil), d0...)
		mulAddRows32AVX2(ker, a, lda, b, ldb, rows)
		mulAddRows32Ref(ref, a, lda, b, ldb, rows)
		for j := range ker {
			if d := float64(ker[j]) - float64(ref[j]); d != 0 && abs(d) > 2*mulRows32Bound(d0[j], a, lda, b, ldb, rows, j) {
				t.Fatalf("AVX2 kernel n=%d rows=%d element %d: %v, portable %v", n, rows, j, ker[j], ref[j])
			}
		}
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
