//go:build amd64

package floats

import (
	"fmt"
	"testing"
)

// BenchmarkBinop32Variants times the AVX2 and SSE2 kernels head to head across
// the sizes from L1-resident to memory-bound, to place the dispatcher's
// crossover (binop32SSEFrom).
func BenchmarkBinop32Variants(b *testing.B) {
	if !hasAVX2FMA {
		b.Skip("no AVX2+FMA")
	}
	type kern struct {
		name string
		f    func(dst, a, b []float32)
	}
	kerns := []kern{
		{"Add32/avx2", add32AVX2}, {"Add32/sse2", add32SSE2},
		{"Mul32/avx2", mul32AVX2}, {"Mul32/sse2", mul32SSE2},
		{"Div32/avx2", div32AVX2}, {"Div32/sse2", div32SSE2},
	}
	for _, n := range []int{1_000, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 1_000_000} {
		a, bb, dst := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range a {
			a[i], bb[i] = 0.25+float32(i%97)*0.01, 0.5+float32(i%89)*0.02
		}
		for _, k := range kerns {
			b.Run(fmt.Sprintf("%s/n=%d", k.name, n), func(b *testing.B) {
				b.SetBytes(int64(12 * n))
				for i := 0; i < b.N; i++ {
					k.f(dst, a, bb)
				}
			})
		}
	}
}
