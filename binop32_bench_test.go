package floats

import (
	"fmt"
	"testing"
)

// BenchmarkBinop32 times each float32 kernel against the scalar Go loop (the
// portable reference kernel) on 1 000 and 100 000 elements:
//
//	go test -run '^$' -bench Binop32 -cpu 1 -count 10 .
//
// Run it at a 1-minute load below 8 only; allocation counts (always 0) are exact
// under load, times are not.
func BenchmarkBinop32(b *testing.B) {
	for _, n := range []int{1_000, 100_000} {
		a, bb, dst := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range a {
			a[i], bb[i] = 0.25+float32(i%97)*0.01, 0.5+float32(i%89)*0.02
		}
		for _, op := range binop32Ops {
			b.Run(fmt.Sprintf("%s/kernel/%d", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(12 * n))
				for i := 0; i < b.N; i++ {
					op.f(dst, a, bb)
				}
			})
			b.Run(fmt.Sprintf("%s/scalar/%d", op.name, n), func(b *testing.B) {
				b.SetBytes(int64(12 * n))
				for i := 0; i < b.N; i++ {
					op.ref(dst, a, bb)
				}
			})
		}
		b.Run(fmt.Sprintf("Sqrt32/kernel/%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				Sqrt32(dst, a)
			}
		})
		b.Run(fmt.Sprintf("Sqrt32/scalar/%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				sqrt32Ref(dst, a)
			}
		})
	}
}
