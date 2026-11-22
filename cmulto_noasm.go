//go:build !amd64 && !arm64 && !s390x && !ppc64le

package floats

// No CMulTo kernel on this architecture: the portable loop of cmulto.go.

func cmulTo(dst, a, b []complex128) { cmulToTail(dst, a, b, 0) }
