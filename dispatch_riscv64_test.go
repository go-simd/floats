//go:build riscv64

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchRISCV64 drives every reduction down both riscv64 paths — the RVV
// kernel and the scalar-reference fallback — by toggling hasRVV. The RVV branch
// is only forced on when the CPU actually has the V extension (the vector
// instructions would trap otherwise); the scalar fallback is always safe. The
// qemu-riscv64 CI job runs with v=true, so both branches are covered there.
func TestDispatchRISCV64(t *testing.T) {
	saved := hasRVV
	defer func() { hasRVV = saved }()

	rng := rand.New(rand.NewSource(99))
	szs := []int{0, 1, 2, 3, 4, 7, 8, 9, 16, 33, 100, 257, 1024}
	check := func(label string) {
		for _, n := range szs {
			a, b := randF64(n, rng), randF64(n, rng)
			if got, want := dot(a, b), dotLanes(a, b); !closeRel(got, want, relTol64) {
				t.Fatalf("%s dot n=%d: %v != %v", label, n, got, want)
			}
			if got, want := sum(a), sumLanes(a); !closeRel(got, want, relTol64) {
				t.Fatalf("%s sum n=%d: %v != %v", label, n, got, want)
			}
			if got, want := sumSqDiff(a, b), sumSqDiffLanes(a, b); !closeRel(got, want, relTol64) {
				t.Fatalf("%s ssd n=%d: %v != %v", label, n, got, want)
			}
			af, bf := randF32(n, rng), randF32(n, rng)
			if got, want := float64(dot32(af, bf)), float64(dotLanes32(af, bf)); !closeRel(got, want, relTol32) {
				t.Fatalf("%s dot32 n=%d: %v != %v", label, n, got, want)
			}
			if got, want := float64(sum32(af)), float64(sumLanes32(af)); !closeRel(got, want, relTol32) {
				t.Fatalf("%s sum32 n=%d: %v != %v", label, n, got, want)
			}
			if got, want := float64(sumSqDiff32(af, bf)), float64(sumSqDiffLanes32(af, bf)); !closeRel(got, want, relTol32) {
				t.Fatalf("%s ssd32 n=%d: %v != %v", label, n, got, want)
			}
		}
	}

	hasRVV = false
	check("scalar")
	if saved {
		hasRVV = true
		check("rvv")
	} else {
		t.Log("CPU lacks RVV; kernel branch not exercised on this host")
	}
}
