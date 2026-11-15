//go:build !arm64 && !amd64

package floats

// No CDotC kernel on this architecture: the portable loop of cdot.go.

func cdotc(a, b []float64) (re, im float64) { return cdotcTail(a, b, 0) }
