//go:build !arm64 && !amd64

package floats

// No squared-magnitude kernel on this architecture: the portable loop of sqmag.go.

func squaredMagnitude(dst, c []float64, s float64) { sqmagTail(dst, c, s, 0) }
