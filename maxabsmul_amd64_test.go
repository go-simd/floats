//go:build amd64

package floats

import (
	"math"
	"math/rand"
	"testing"
)

// The AVX2 kernel itself (not only through MaxAbsMul) agrees with the fmax loop
// in the value of the maximum; only the sign of a zero maximum is left to the
// wrapper. Skipped, and said so, on a CPU without AVX2+FMA.
func TestMaxAbsMulAVX2Direct(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA: the dispatcher uses the portable loop")
	}
	rng := rand.New(rand.NewSource(3))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 80; trial++ {
			v, w := make([]float64, n), make([]float64, n)
			for i := range v {
				v[i], w[i] = mamSpecial(rng), mamSpecial(rng)
			}
			want, got := vmnormRef(v, w), maxAbsMulAVX2(v, w)
			if want == 0 {
				if got != 0 {
					t.Fatalf("n=%d: kernel %v, loop zero", n, got)
				}
				continue
			}
			if math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("n=%d: kernel %v (%#x), loop %v (%#x)", n, got, math.Float64bits(got), want, math.Float64bits(want))
			}
		}
	}
}
