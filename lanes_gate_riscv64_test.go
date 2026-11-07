//go:build riscv64

package floats

import "testing"

// TestLanesFallbackRISCV64 runs the elementwise, Sqrt and pairwise-block suites with the
// hasRVV gate off, so the portable fallback branches of lanes_riscv64.go are covered
// (and checked) on a CPU that has the kernels.
func TestLanesFallbackRISCV64(t *testing.T) {
	saved := hasRVV
	defer func() { hasRVV = saved }()
	hasRVV = false
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
