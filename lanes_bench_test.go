package floats

import (
	"fmt"
	"testing"
)

// BenchmarkLanes pits the dispatched elementwise kernels and SumPairwise against
// their portable references (the dispatch rule of this library: a kernel stays
// only where it is measured faster on real silicon). Interleave the runs
// (-count) and never read a number taken under load.
func BenchmarkLanes(b *testing.B) {
	for _, n := range []int{64, 1024, 65536} {
		x, y, d := make([]float64, n), make([]float64, n), make([]float64, n)
		for i := range x {
			x[i], y[i] = float64(i)+1, float64(n-i)
		}
		for _, c := range []struct {
			name string
			fn   func()
		}{
			{"Add/kernel", func() { add(d, x, y) }},
			{"Add/ref", func() { addRef(d, x, y) }},
			{"Mul/kernel", func() { mul(d, x, y) }},
			{"Mul/ref", func() { mulRef(d, x, y) }},
			{"Div/kernel", func() { div(d, x, y) }},
			{"Div/ref", func() { divRef(d, x, y) }},
			{"Sqrt/kernel", func() { sqrt(d, x) }},
			{"Sqrt/ref", func() { sqrtRef(d, x) }},
			{"SumPairwise/kernel", func() { sinkF = SumPairwise(x) }},
			{"SumPairwise/ref", func() { sinkF = pwInit + pairwiseRef(x) }},
		} {
			b.Run(fmt.Sprintf("%s/n=%d", c.name, n), func(b *testing.B) {
				b.SetBytes(int64(n) * 8)
				for i := 0; i < b.N; i++ {
					c.fn()
				}
			})
		}
	}
}

var sinkF float64
