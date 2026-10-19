package floats

import (
	"fmt"
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

// ---- elementwise updates vs the scalar loops callers write by hand ----

func elemSizes() []int { return []int{1000, 100000, 1000000} }

func naiveAxpy(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func naiveXpby(x []float64, beta float64, y []float64) {
	for i, v := range x {
		y[i] = v + beta*y[i]
	}
}

func naiveAxpyDot(alpha float64, x, y, w []float64) float64 {
	var s float64
	for i, v := range x {
		y[i] += alpha * v
		s += y[i] * w[i]
	}
	return s
}

func naiveScale(alpha float64, x []float64) {
	for i := range x {
		x[i] *= alpha
	}
}

// benchElem reports bytes moved per call (arrays read + written) so -bench
// prints a memory bandwidth comparable across sizes. alpha alternates sign
// between calls so the data stays bounded over millions of iterations.
func benchElem(b *testing.B, arrays int, simd, naive func(x, y, w []float64, alpha float64)) {
	rng := rand.New(rand.NewSource(5))
	for _, n := range elemSizes() {
		x, y, w := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		for _, c := range []struct {
			name string
			f    func(x, y, w []float64, alpha float64)
		}{{"simd", simd}, {"naive", naive}} {
			b.Run(fmt.Sprintf("%d/%s", n, c.name), func(b *testing.B) {
				b.SetBytes(int64(n * 8 * arrays))
				alpha := 1e-3
				for i := 0; i < b.N; i++ {
					c.f(x, y, w, alpha)
					alpha = -alpha
				}
			})
		}
	}
}

func BenchmarkAxpy(b *testing.B) {
	benchElem(b, 3,
		func(x, y, _ []float64, a float64) { Axpy(a, x, y) },
		func(x, y, _ []float64, a float64) { naiveAxpy(a, x, y) })
}

func BenchmarkXpby(b *testing.B) {
	benchElem(b, 3,
		func(x, y, _ []float64, a float64) { Xpby(x, a, y) },
		func(x, y, _ []float64, a float64) { naiveXpby(x, a, y) })
}

var sink float64

func BenchmarkAxpyDot(b *testing.B) {
	benchElem(b, 4,
		func(x, y, w []float64, a float64) { sink = AxpyDot(a, x, y, w) },
		func(x, y, w []float64, a float64) { sink = naiveAxpyDot(a, x, y, w) })
}

func BenchmarkScale(b *testing.B) {
	benchElem(b, 2,
		func(x, _, _ []float64, a float64) { Scale(1+a, x) },
		func(x, _, _ []float64, a float64) { naiveScale(1+a, x) })
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
