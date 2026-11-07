//go:build ppc64le

package floats

import "testing"

// TestLanesFallbackPPC64LE runs the elementwise, Sqrt and pairwise-block suites with the
// hasVSX gate off, so the portable fallback branches of lanes_ppc64le.go are covered
// (and checked) on a CPU that has the kernels.
func TestLanesFallbackPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()
	hasVSX = false
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
