//go:build loong64

package floats

import "testing"

// TestRoundFallbackLoong64 runs the Abs ... RoundEven suites with the hasLASX gate
// off, so the portable fallback branches of round_loong64.go are covered (and
// checked) on a CPU that has the kernels.
func TestRoundFallbackLoong64(t *testing.T) {
	saved := hasLASX
	defer func() { hasLASX = saved }()
	hasLASX = false
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
