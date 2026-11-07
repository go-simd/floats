//go:build !amd64 && !arm64

package floats

// No kernel on these architectures yet: the Go loops (expCR, logCR, Log). A
// kernel is added when it is measured faster on real silicon (the dispatch
// rule of this library).

// exp32Fast is the correctly rounded float32 exp: expCR in Go.
func exp32Fast(dst, src []float32) { exp32CR(dst, src) }

// log32Fast is the correctly rounded float32 log: logCR in Go.
func log32Fast(dst, src []float32) { logCRSlice(dst, src) }

// logFast is Log's Go loop.
func logFast(dst, src []float64) { logScalar(dst, src) }
