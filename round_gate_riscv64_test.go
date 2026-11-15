//go:build riscv64

package floats

import "testing"

// TestRoundFallbackRISCV64 runs the Abs ... RoundEven suites with the hasRVV gate
// off, so the portable fallback branches of round_riscv64.go are covered (and
// checked) on a CPU that has the kernels.
func TestRoundFallbackRISCV64(t *testing.T) {
	saved := hasRVV
	defer func() { hasRVV = saved }()
	hasRVV = false
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
