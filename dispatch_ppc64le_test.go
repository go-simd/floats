//go:build ppc64le

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchPPC64LE drives both ppc64le dispatch paths to 100% coverage.
//
// The float64 reductions have no VSX kernel on ppc64le — on real POWER9 the
// gc-autovectorized naive loop is faster (see kernels_ppc64le.go) — so they run
// unconditionally and are covered by the package's normal differential tests;
// they are exercised here too for completeness against the lane-blocked oracle.
//
// The float32 reductions keep the VSX kernel behind hasVSX. This test toggles
// hasVSX (restoring it with defer) to cover BOTH the VSX branch and the naive
// f32 fallback. The kernel emits only ISA-2.06 VSX ops (the ppc64le/POWER8
// baseline), so it does not SIGILL on POWER8 and both branches are forced
// unconditionally here. The QEMU power8 and power9 CI jobs plus the native
// POWER8E/POWER9 farm runs all cover the kernel branch.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	rng := rand.New(rand.NewSource(99))
	szs := []int{0, 1, 2, 3, 4, 7, 8, 9, 16, 33, 100, 257, 1024}

	// float64: single unconditional path, checked against the lane-blocked oracle.
	for _, n := range szs {
		a, b := randF64(n, rng), randF64(n, rng)
		if got, want := dot(a, b), dotLanes(a, b); !closeDot(got, want, absSumProd64(a, b), relTol64) {
			t.Fatalf("f64 dot n=%d: %v != %v", n, got, want)
		}
		if got, want := sum(a), sumLanes(a); !closeDot(got, want, absSum64(a), relTol64) {
			t.Fatalf("f64 sum n=%d: %v != %v", n, got, want)
		}
		if got, want := sumSqDiff(a, b), sumSqDiffLanes(a, b); !closeRel(got, want, relTol64) {
			t.Fatalf("f64 ssd n=%d: %v != %v", n, got, want)
		}
	}

	// float32: drive both the VSX kernel and the naive fallback.
	checkF32 := func(label string) {
		for _, n := range szs {
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

	// Naive f32 fallback: always safe.
	hasVSX = false
	checkF32("f32 fallback")

	// VSX f32 kernel: ISA-2.06 ops only (POWER8 baseline), so force it on
	// unconditionally — it runs on every ppc64le host.
	hasVSX = true
	checkF32("f32 kernel")
}
