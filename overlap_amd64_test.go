//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestOverlapWindowPortableForced: the path of a CPU without AVX2+FMA, forced on
// any host, held to the same bound.
func TestOverlapWindowPortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	rng := rand.New(rand.NewSource(36))
	for _, n := range []int{1, 7, 8, 9, 64, 129} {
		a, b, win := overlapInputs(rng, n)
		dst := make([]float64, 2*n)
		OverlapWindow(dst, a, b, win)
		checkBound(t, "OverlapWindow portable", n, dst, a, b, win, 53)
		a32, b32, w32 := to32(a), to32(b), to32(win)
		d32 := make([]float32, 2*n)
		Float32OverlapWindow(d32, a32, b32, w32)
		checkBound(t, "Float32OverlapWindow portable", n, to64(d32), to64(a32), to64(b32), to64(w32), 24)
	}
}

// TestOverlapWindowAVX2Dispatch: on a CPU with AVX2+FMA the public functions run the
// kernels (and say so); the kernels are also held to the bound directly, so the
// test cannot pass by quietly taking the portable path.
func TestOverlapWindowAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestOverlapWindowBound")
	}
	rng := rand.New(rand.NewSource(35))
	for _, n := range overlapSizes {
		a, b, win := overlapInputs(rng, n)
		dst := make([]float64, 2*n)
		overlapWindow64AVX2(dst, a, b, win)
		overlapTail64(dst, a, b, win, n&^3)
		checkBound(t, "overlapWindow64AVX2", n, dst, a, b, win, 53)
		a32, b32, w32 := to32(a), to32(b), to32(win)
		d32 := make([]float32, 2*n)
		overlapWindow32AVX2(d32, a32, b32, w32)
		overlapTail32(d32, a32, b32, w32, n&^7)
		checkBound(t, "overlapWindow32AVX2", n, to64(d32), to64(a32), to64(b32), to64(w32), 24)
	}
}
