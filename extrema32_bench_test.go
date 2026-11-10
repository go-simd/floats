package floats

import (
	"fmt"
	"testing"
)

// naiveMax32 is the loop Float32Max was before the kernels: one chain of Go's max.
func naiveMax32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		m = max(m, v)
	}
	return m
}

func naiveMin32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		m = min(m, v)
	}
	return m
}

var sinkF32 float32

// BenchmarkFloat32Extrema compares the dispatched Float32Max/Float32Min with the
// single-chain loop (naive) and the four-chain loop (ref).
func BenchmarkFloat32Extrema(b *testing.B) {
	for _, n := range []int{16, 1000, 100000, 1000000} {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(i%1021) - 500
		}
		for _, c := range []struct {
			name string
			fn   func([]float32) float32
		}{
			{"Max", Float32Max}, {"MaxNaive", naiveMax32}, {"MaxRef", maxRef32},
			{"Min", Float32Min}, {"MinNaive", naiveMin32}, {"MinRef", minRef32},
		} {
			b.Run(fmt.Sprintf("%s/%d", c.name, n), func(b *testing.B) {
				b.SetBytes(int64(4 * n))
				for i := 0; i < b.N; i++ {
					sinkF32 = c.fn(a)
				}
			})
		}
	}
}
