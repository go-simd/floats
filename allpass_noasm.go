//go:build !arm64 && !amd64

package floats

// No all-pass kernel on this architecture: the portable math.FMA loop of allpass.go.

func allPass3(n int, out, in []complex128, t *AllPassRows, old, cur *[3][]complex128) {
	allPass3Ref(n, out, in, t, old, cur)
}
