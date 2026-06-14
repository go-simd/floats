package floats

import (
	"math/rand"
	"testing"
)

// naiveDot64 is the textbook left-to-right scalar dot product — the baseline a
// hand-rolled Go loop (and gonum's pure-Go fallback) would use.
func naiveDot64(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func benchSizes() []int { return []int{8, 64, 512, 4096} }

func BenchmarkDot(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range benchSizes() {
		x, y := randF64(n, rng), randF64(n, rng)
		b.Run(sizeName(n)+"/simd", func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = Dot(x, y)
			}
		})
		b.Run(sizeName(n)+"/naive", func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = naiveDot64(x, y)
			}
		})
	}
}

func BenchmarkFloat32Dot(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range benchSizes() {
		x, y := randF32(n, rng), randF32(n, rng)
		b.Run(sizeName(n), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for i := 0; i < b.N; i++ {
				_ = Float32Dot(x, y)
			}
		})
	}
}

func BenchmarkDistance(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range benchSizes() {
		x, y := randF64(n, rng), randF64(n, rng)
		b.Run(sizeName(n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = Distance(x, y)
			}
		})
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	for _, n := range benchSizes() {
		x, y := randF32(n, rng), randF32(n, rng)
		b.Run(sizeName(n), func(b *testing.B) {
			b.SetBytes(int64(n * 4))
			for i := 0; i < b.N; i++ {
				_ = Float32CosineSimilarity(x, y)
			}
		})
	}
}

func sizeName(n int) string {
	switch n {
	case 8:
		return "8"
	case 64:
		return "64"
	case 512:
		return "512"
	case 4096:
		return "4096"
	}
	return "?"
}
