//go:build !arm64 && !amd64

package floats

// No conjugate-reverse kernel on this architecture: the portable loop of conjrev.go.

func conjReverse(dst, src []float64) { conjRevTail(dst, src, 0) }
