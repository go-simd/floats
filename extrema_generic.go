//go:build !amd64 && !arm64

package floats

// No Max/Min kernel on this architecture: the loop.

func maxF64(a []float64) float64 { return maxRef(a) }
func minF64(a []float64) float64 { return minRef(a) }
