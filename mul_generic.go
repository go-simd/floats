//go:build !amd64 && !arm64 && !s390x && !ppc64le && !loong64 && !riscv64

package floats

// Mul and the pairwise-sum block on an architecture with no kernel: the portable
// loops, which are the reference the kernels of the other architectures are
// tested against.

func mul(dst, a, b []float64)           { mulRef(dst, a, b) }
func pairwiseBlock(a []float64) float64 { return pairwiseBlockRef(a) }
