//go:build !arm64 && !amd64

package floats

// No kernel on this architecture: the portable fold.
func maxAbsMul(v, w []float64) float64 { return maxAbsMulRef(v, w) }
