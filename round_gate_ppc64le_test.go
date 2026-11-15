//go:build ppc64le

package floats

import "testing"

// TestRoundFallbackPPC64LE runs the Abs ... RoundEven suites with the hasVSX gate
// off, so the portable fallback branches of round_ppc64le.go are covered (and
// checked) on a CPU that has the kernels.
func TestRoundFallbackPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()
	hasVSX = false
	for _, tc := range []struct {
		name string
		fn   func(*testing.T)
	}{
		{"EdgeTable", TestRoundEdgeTable},
		{"Random", TestRoundRandomDifferential},
		{"Aliasing", TestRoundAliasing},
	} {
		t.Run(tc.name, tc.fn)
	}
}
