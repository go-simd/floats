//go:build ppc64le

package floats

import "testing"

// TestPairwiseFallbackPPC64LE runs the Maximum / Minimum suites with the hasVSX
// gate off, so the portable fallback branches of maxmin_ppc64le.go are covered (and
// checked) on a CPU that has the kernels; the gate it found is logged.
func TestPairwiseFallbackPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()
	t.Logf("hasVSX on this CPU: %v", saved)
	hasVSX = false
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
