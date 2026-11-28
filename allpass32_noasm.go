//go:build !arm64 && !amd64

package floats

// No split-plane all-pass kernel on this architecture: the portable loop of allpass32.go.

func allPass3Split32(n int, out, in Split32, t *AllPassRows32, old, cur *[3]Split32) {
	allPass3Split32Ref(n, out, in, t, old, cur)
}
