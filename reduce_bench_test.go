package floats

import (
	"fmt"
	"math/rand"
	"testing"
)

func redBenchSizes() []int { return []int{3, 8, 12, 16, 24, 32, 64, 256, 1000, 4096, 100000, 1000000} }

func BenchmarkReduce(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range redBenchSizes() {
		x, y := randF64(n, rng), randF64(n, rng)
		xf, yf := randF32(n, rng), randF32(n, rng)
		b.Run(fmt.Sprintf("Dot/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(16 * n))
			for i := 0; i < b.N; i++ {
				_ = Dot(x, y)
			}
		})
		b.Run(fmt.Sprintf("Sum/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				_ = Sum(x)
			}
		})
		b.Run(fmt.Sprintf("Distance/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(16 * n))
			for i := 0; i < b.N; i++ {
				_ = Distance(x, y)
			}
		})
		b.Run(fmt.Sprintf("Float32Dot/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				_ = Float32Dot(xf, yf)
			}
		})
		b.Run(fmt.Sprintf("Float32Sum/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				_ = Float32Sum(xf)
			}
		})
	}
}
