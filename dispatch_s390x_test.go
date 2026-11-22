//go:build s390x

package floats

import (
	"math/rand"
	"testing"
)

// TestDispatchS390X drives both s390x dispatch paths of the reductions: the
// float64 plain loops below vxMin64 and the VX kernels from it up, and the
// float32 VX kernels (vector-enhancements facility 1, gated on hasVXE) against
// the lane-blocked scalar fallback. Sizes straddle vxMin64 and the 8-register
// block (16 f64, 32 f32 elements) so the bulk, one-register and scalar loops all
// run, with the error bound as the contract (not bit equality).
func TestDispatchS390X(t *testing.T) {
	saved := hasVXE
	defer func() { hasVXE = saved }()

	rng := rand.New(rand.NewSource(77))
	szs := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257, 1024, 4099}

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

	hasVXE = false
	checkF32("f32 fallback")
	if saved {
		hasVXE = true
		checkF32("f32 kernel")
	}
}
