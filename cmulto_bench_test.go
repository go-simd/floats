package floats

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkCMulTo times CMulTo against the scalar Go loop (cmulToTail from 0: Go's
// complex product) over sizes from L1 to beyond L2. -benchmem is the allocation check.
func BenchmarkCMulTo(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{16, 64, 256, 1024, 4096, 65536} {
		x, y, d := cmulRand(rng, n, 1), cmulRand(rng, n, 1), make([]complex128, n)
		b.Run(fmt.Sprintf("kernel/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(48 * n))
			for i := 0; i < b.N; i++ {
				CMulTo(d, x, y)
			}
		})
		b.Run(fmt.Sprintf("scalar/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(48 * n))
			for i := 0; i < b.N; i++ {
				cmulToTail(d, x, y, 0)
			}
		})
	}
}
