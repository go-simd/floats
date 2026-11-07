//go:build loong64

package floats

import "testing"

// TestLanesFallbackLoong64 runs the elementwise, Sqrt and pairwise-block suites with the
// hasLASX gate off, so the portable fallback branches of lanes_loong64.go are covered
// (and checked) on a CPU that has the kernels.
func TestLanesFallbackLoong64(t *testing.T) {
	saved := hasLASX
	defer func() { hasLASX = saved }()
	hasLASX = false
	for _, tc := range []struct {
		name string
		fn   func(*testing.T)
	}{
		{"Binop", TestBinopBitIdentical},
		{"Sqrt", TestSqrtBitIdentical},
		{"Mul", TestMulBitIdentical},
		{"PairwiseBlock", TestPairwiseBlockAllLengths},
		{"PairwiseSIMD", TestSumPairwiseSIMDMatchesReference},
	} {
		t.Run(tc.name, tc.fn)
	}
}
