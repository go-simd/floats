package floats

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkRound times every function of round.go against the scalar Go loop (the
// portable kernel, which is what go-ndarray ran) on the same data, at a size that
// fits the L1 cache (1e3) and one that fits the L2 (1e5 is 800 kB: L2/L3).
func BenchmarkRound(b *testing.B) {
	rng := rand.New(rand.NewSource(74))
	for _, n := range []int{1000, 100000} {
		src := make([]float64, n)
		for i := range src {
			src[i] = rng.NormFloat64() * 1000
		}
		dst := make([]float64, n)
		for _, op := range roundOps {
			b.Run(fmt.Sprintf("%s/%d/kernel", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(8 * n))
				for i := 0; i < b.N; i++ {
					op.f(dst, src)
				}
			})
			b.Run(fmt.Sprintf("%s/%d/loop", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(8 * n))
				for i := 0; i < b.N; i++ {
					op.ref(dst, src)
				}
			})
		}
	}
}
