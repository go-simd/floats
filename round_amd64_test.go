//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64Round drives Abs, Neg, Square, Floor, Ceil, Trunc and
// RoundEven down both amd64 paths, the AVX2 kernels and the fallback (the SSE2
// kernels for the first three, the scalar loop for the rounding functions), by
// toggling hasAVX2FMA: every result is the Go expression's, bit for bit.
func TestDispatchAMD64Round(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	edges := roundEdges()
	rng := rand.New(rand.NewSource(98))
	check := func(label string) {
		for _, op := range roundOps {
			for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 33, 100, 257} {
				for off := 0; off < 4; off++ {
					src, _ := randomWords(n, off, rng, true)
					for i := range src {
						if i%3 == 0 {
							src[i] = edges[(i*7+n)%len(edges)]
						}
					}
					dst, buf := fillE(n, (off+1)%4, rng, false)
					op.f(dst, src)
					checkRound(t, op, label, dst, src, n, off)
					checkGuard(t, op.name, buf, n, (off+1)%4)
				}
			}
		}
	}
	hasAVX2FMA = false
	check("fallback")
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; kernel branch not exercised on this host")
	}
}
