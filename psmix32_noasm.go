//go:build !arm64 && !amd64

package floats

// No split-plane PS mixing kernel on this architecture: the portable loop of psmix32.go.

func psMix32(p *PSMixPlan32, gains, tg []float32, s Split32, d *[3]Split32, l, r Split32) {
	psMix32Ref(p, gains, tg, s, d, l, r)
}
