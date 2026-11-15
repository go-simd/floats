package floats

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkPairwise times Maximum and Minimum against the scalar Go loop ndarray
// ran (dst[i] = max(a[i], b[i]), the Go builtin) on the same data, at a size that
// fits the L1 cache (1e3) and one that fits the L2/L3 (1e5 is 800 kB per slice).
// The "ref" rows time the portable maximumRef, which spells the NaN rule out.
func BenchmarkPairwise(b *testing.B) {
	rng := rand.New(rand.NewSource(75))
	for _, n := range []int{1000, 100000} {
		x, y, dst := make([]float64, n), make([]float64, n), make([]float64, n)
		for i := range x {
			x[i], y[i] = rng.NormFloat64()*1000, rng.NormFloat64()*1000
		}
		loops := map[string]func(){
			"Maximum": func() {
				for i := range dst {
					dst[i] = max(x[i], y[i])
				}
			},
			"Minimum": func() {
				for i := range dst {
					dst[i] = min(x[i], y[i])
				}
			},
		}
		for _, op := range maxminOps {
			b.Run(fmt.Sprintf("%s/%d/kernel", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(16 * n))
				for i := 0; i < b.N; i++ {
					op.f(dst, x, y)
				}
			})
			b.Run(fmt.Sprintf("%s/%d/loop", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(16 * n))
				for i := 0; i < b.N; i++ {
					loops[op.name]()
				}
			})
			b.Run(fmt.Sprintf("%s/%d/ref", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(16 * n))
				for i := 0; i < b.N; i++ {
					op.ref(dst, x, y)
				}
			})
		}
	}
}
