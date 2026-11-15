//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64Pairwise drives Maximum and Minimum down both amd64 paths, the
// AVX2 kernels and the SSE2 kernels, by toggling hasAVX2FMA: every result is the
// rule's, bit for bit, on the edge pairs placed at every position of every length
// and alignment, and on random bit patterns.
func TestDispatchAMD64Pairwise(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	edges := maxminEdges()
	rng := rand.New(rand.NewSource(98))
	check := func(label string) {
		for _, op := range maxminOps {
			for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 33, 100, 257} {
				for off := 0; off < 4; off++ {
					a, b := pairWords(n, off, rng)
					for i := range a {
						if i%3 == 0 {
							a[i], b[i] = edges[(i*7+n)%len(edges)], edges[(i*5+n+off)%len(edges)]
						}
					}
					dst, buf := fillE(n, (off+2)%4, rng, false)
					op.f(dst, a, b)
					checkMaxMin(t, op, label, dst, a, b, n, off)
					checkGuard(t, op.name, buf, n, (off+2)%4)
				}
			}
		}
	}
	hasAVX2FMA = false
	check("sse2")
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; kernel branch not exercised on this host")
	}
}
