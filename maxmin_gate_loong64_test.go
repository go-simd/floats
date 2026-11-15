//go:build loong64

package floats

import "testing"

// TestPairwiseFallbackLoong64 runs the Maximum / Minimum suites with the hasLASX
// gate off, so the portable fallback branches of maxmin_loong64.go are covered (and
// checked) on a CPU that has the kernels; the gate it found is logged.
func TestPairwiseFallbackLoong64(t *testing.T) {
	saved := hasLASX
	defer func() { hasLASX = saved }()
	t.Logf("hasLASX on this CPU: %v", saved)
	hasLASX = false
	pairwiseFallback = true
	defer func() { pairwiseFallback = false }()
	for _, tc := range []struct {
		name string
		fn   func(*testing.T)
	}{
		{"EdgeTable", TestPairwiseEdgeTable},
		{"Random", TestPairwiseRandomDifferential},
		{"Aliasing", TestPairwiseAliasing},
	} {
		t.Run(tc.name, tc.fn)
	}
}
