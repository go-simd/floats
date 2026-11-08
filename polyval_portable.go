//go:build !amd64 && !arm64

package floats

// ppc64le, s390x, riscv64, loong64 and the 32-bit targets have no Horner or
// Clenshaw kernel yet (floats#30): the portable loops are the kernels, with the
// same bits.

func polyvalKernel(dst, x, c []float64) { polyvalRef(dst, x, c) }
func chebvalKernel(dst, x, c []float64) { chebvalRef(dst, x, c) }
