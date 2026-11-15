//go:build !amd64 && !arm64 && !s390x && !ppc64le && !loong64 && !riscv64

package floats

// No kernel on this architecture: the portable loops. amd64, arm64, s390x,
// ppc64le, loong64 and riscv64 have kernels (maxmin_<arch>.go).

func vmaximum(dst, a, b []float64) { maximumRef(dst, a, b) }
func vminimum(dst, a, b []float64) { minimumRef(dst, a, b) }
