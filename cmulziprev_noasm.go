//go:build !amd64

package floats

// No CMulZipRev kernel on this architecture: the portable loop of cmulziprev.go.

func cmulZipRev(dst, x, y, w []float64) { zipRevTail(dst, x, y, w, 0) }
