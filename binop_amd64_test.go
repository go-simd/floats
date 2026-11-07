//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64Binop drives the kernels down both amd64 paths, the AVX2
// kernel and the portable fallback, by toggling hasAVX2FMA: bit for bit equal.
func TestDispatchAMD64Binop(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	rng := rand.New(rand.NewSource(97))
	check := func(label string) {
		for _, n := range []int{0, 1, 3, 4, 5, 15, 16, 17, 33, 100, 257} {
			a, _ := fillE(n, 0, rng, true)
			b, _ := fillE(n, 1, rng, true)
			got, want := make([]float64, n), make([]float64, n)
			addRef(want, a, b)
			Add(got, a, b)
			checkSame(t, label+" Add", got, want, n, 0)
			subRef(want, a, b)
			Sub(got, a, b)
			checkSame(t, label+" Sub", got, want, n, 0)
			divRef(want, a, b)
			Div(got, a, b)
			checkSame(t, label+" Div", got, want, n, 0)
			sqrtRef(want, a)
			Sqrt(got, a)
			checkSame(t, label+" Sqrt", got, want, n, 0)
			clampScalarRef(want, a, -0.5, 0.5)
			ClampScalar(got, a, -0.5, 0.5)
			exactBits(t, label+" ClampScalar", got, want, n, 0)
		}
	}
	hasAVX2FMA = false
	check("portable")
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; kernel branch not exercised on this host")
	}
}
