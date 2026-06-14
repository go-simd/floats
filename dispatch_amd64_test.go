//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchAMD64 drives every reduction down BOTH amd64 paths — the AVX2+FMA
// kernel and the scalar-reference fallback — by toggling hasAVX2FMA, restoring
// it with defer. The AVX2 branch is only forced on when the CPU actually has
// AVX2+FMA (the VFMADD231* instructions would #UD otherwise); the scalar
// fallback is always safe. The native amd64 CI runner has AVX2+FMA, so both
// branches are covered there, making it the authoritative 100%-coverage gate for
// this file's six dispatchers.
func TestDispatchAMD64(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()

	rng := rand.New(rand.NewSource(99))
	szs := []int{0, 1, 2, 3, 4, 7, 8, 9, 16, 33, 100, 257, 1024}
	check := func(label string) {
		for _, n := range szs {
			a, b := randF64(n, rng), randF64(n, rng)
			if got, want := dot(a, b), dotLanes(a, b); !closeDot(got, want, absSumProd64(a, b), relTol64) {
				t.Fatalf("%s dot n=%d: %v != %v", label, n, got, want)
			}
			if got, want := sum(a), sumLanes(a); !closeDot(got, want, absSum64(a), relTol64) {
				t.Fatalf("%s sum n=%d: %v != %v", label, n, got, want)
			}
			if got, want := sumSqDiff(a, b), sumSqDiffLanes(a, b); !closeRel(got, want, relTol64) {
				t.Fatalf("%s ssd n=%d: %v != %v", label, n, got, want)
			}
			af, bf := randF32(n, rng), randF32(n, rng)
			if got, want := float64(dot32(af, bf)), float64(dotLanes32(af, bf)); !closeDot(got, want, absSumProd32(af, bf), relTol32) {
				t.Fatalf("%s dot32 n=%d: %v != %v", label, n, got, want)
			}
			if got, want := float64(sum32(af)), float64(sumLanes32(af)); !closeDot(got, want, absSum32(af), relTol32) {
				t.Fatalf("%s sum32 n=%d: %v != %v", label, n, got, want)
			}
			if got, want := float64(sumSqDiff32(af, bf)), float64(sumSqDiffLanes32(af, bf)); !closeRel(got, want, relTol32) {
				t.Fatalf("%s ssd32 n=%d: %v != %v", label, n, got, want)
			}
		}
	}

	// Scalar fallback: always safe.
	hasAVX2FMA = false
	check("scalar")

	// AVX2+FMA kernel: only if the host has it.
	if saved {
		hasAVX2FMA = true
		check("avx2")
	} else {
		t.Log("CPU lacks AVX2+FMA; kernel branch not exercised on this host")
	}
}
