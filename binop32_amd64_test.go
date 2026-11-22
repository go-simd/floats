//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64Binop32 drives the float32 kernels down both amd64 paths, the
// AVX2 kernels and the SSE2 baseline ones, by toggling hasAVX2FMA: bit for bit
// equal to the portable loops.
func TestDispatchAMD64Binop32(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	rng := rand.New(rand.NewSource(98))
	check := func(label string) {
		for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 257,
			binop32SSEFrom - 1, binop32SSEFrom, binop32SSEFrom + 1, binop32SSEFrom + 37} {
			a, _ := fill32(n, 0, rng, true)
			b, _ := fill32(n, 1, rng, true)
			got, want := make([]float32, n), make([]float32, n)
			for _, op := range binop32Ops {
				op.ref(want, a, b)
				op.f(got, a, b)
				checkSame32(t, label+" "+op.name, got, want, n, 0)
			}
			sqrt32Ref(want, a)
			Sqrt32(got, a)
			checkSame32(t, label+" Sqrt32", got, want, n, 0)
		}
	}
	hasAVX2FMA = false
	check("sse2")
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; AVX2 kernel branch not exercised on this host")
	}
}
