//go:build !arm64

package floats

// No kernel on this architecture: the portable fold (a follow-up for AVX2).
func maxAbsMul(v, w []float64) float64 { return maxAbsMulRef(v, w) }
