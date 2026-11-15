//go:build riscv64

package floats

import "testing"

// TestPairwiseFallbackRISCV64 runs the Maximum / Minimum suites with the hasRVV
// gate off, so the portable fallback branches of maxmin_riscv64.go are covered (and
// checked) on a CPU that has the kernels; the gate it found is logged.
func TestPairwiseFallbackRISCV64(t *testing.T) {
	saved := hasRVV
	defer func() { hasRVV = saved }()
	t.Logf("hasRVV on this CPU: %v", saved)
	hasRVV = false
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
