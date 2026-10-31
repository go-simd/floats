package floats

import (
	"fmt"
	"math"
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

// ---- Mul, Axpby3, Clamp, MaxAbs vs the loops callers write by hand ----

func naiveMul(dst, a, b []float64) {
	for i, v := range a {
		dst[i] = v * b[i]
	}
}

func naiveAxpby3(dst []float64, al float64, a []float64, be float64, b []float64, ga float64, c []float64) {
	for i, v := range a {
		dst[i] = al*v + be*b[i] + ga*c[i]
	}
}

func naiveClamp(dst, v, lo, hi []float64) {
	for i, x := range v {
		dst[i] = math.Min(math.Max(x, lo[i]), hi[i])
	}
}

// naiveClampCmp is the same projection with comparisons (the form Clamp pins).
func naiveClampCmp(dst, v, lo, hi []float64) {
	for i, x := range v {
		if lo[i] > x {
			x = lo[i]
		}
		if hi[i] < x {
			x = hi[i]
		}
		dst[i] = x
	}
}

func naiveMaxAbsLoop(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if a := math.Abs(x); a > m || a != a {
			m = a
		}
	}
	return m
}

// benchRun runs f (simd) and g (naive) at the elemSizes with b.SetBytes(n*8*arrays).
func benchRun(b *testing.B, arrays int, f, g func(n int) func()) {
	for _, n := range elemSizes() {
		for _, c := range []struct {
			name string
			mk   func(n int) func()
		}{{"simd", f}, {"naive", g}} {
			run := c.mk(n)
			b.Run(fmt.Sprintf("%d/%s", n, c.name), func(b *testing.B) {
				b.SetBytes(int64(n * 8 * arrays))
				for i := 0; i < b.N; i++ {
					run()
				}
			})
		}
	}
}

func benchData(n int, k int) [][]float64 {
	rng := rand.New(rand.NewSource(7))
	out := make([][]float64, k)
	for i := range out {
		out[i] = randF64(n, rng)
	}
	return out
}

func BenchmarkMul(b *testing.B) {
	mk := func(f func(d, a, b []float64)) func(n int) func() {
		return func(n int) func() {
			d := benchData(n, 3)
			return func() { f(d[0], d[1], d[2]) }
		}
	}
	benchRun(b, 3, mk(Mul), mk(naiveMul))
}

func BenchmarkAxpby3(b *testing.B) {
	mk := func(f func(d []float64, al float64, a []float64, be float64, b []float64, ga float64, c []float64)) func(n int) func() {
		return func(n int) func() {
			d := benchData(n, 4)
			return func() { f(d[0], 0.5, d[1], -1.5, d[2], 2.5, d[3]) }
		}
	}
	benchRun(b, 4, mk(Axpby3), mk(naiveAxpby3))
}

func benchClamp(b *testing.B, naive func(d, v, lo, hi []float64)) {
	mk := func(f func(d, v, lo, hi []float64)) func(n int) func() {
		return func(n int) func() {
			d := benchData(n, 4)
			for i := range d[2] { // a proper box: lo < hi
				d[2][i], d[3][i] = -math.Abs(d[2][i]), math.Abs(d[3][i])
			}
			return func() { f(d[0], d[1], d[2], d[3]) }
		}
	}
	benchRun(b, 4, mk(Clamp), mk(naive))
}

// BenchmarkClamp compares against the math.Min/math.Max form (NaN-propagating,
// signed-zero aware: the form people write) and BenchmarkClampCmp against the
// comparison form (same semantics as Clamp).
func BenchmarkClamp(b *testing.B)    { benchClamp(b, naiveClamp) }
func BenchmarkClampCmp(b *testing.B) { benchClamp(b, naiveClampCmp) }

func BenchmarkMaxAbs(b *testing.B) {
	mk := func(f func(v []float64) float64) func(n int) func() {
		return func(n int) func() {
			d := benchData(n, 1)
			return func() { sink = f(d[0]) }
		}
	}
	benchRun(b, 1, mk(MaxAbs), mk(naiveMaxAbsLoop))
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
