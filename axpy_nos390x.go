//go:build !amd64 && !arm64 && !s390x

package floats

// axpy on the architectures with no assembly kernel for it: the portable
// math.FMA loop of axpy.go. s390x has its own (kernels_s390x.go).
func axpy(alpha float64, x, y []float64) { axpyRef(alpha, x, y) }
