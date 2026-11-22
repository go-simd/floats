//go:build amd64

package floats

// runExhaustiveKernels runs the exhaustive check down the SSE2 kernel and, when
// the CPU has AVX2+FMA, the AVX2 one.
func runExhaustiveKernels(run func(label string)) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	run("sse2")
	if saved {
		hasAVX2FMA = true
		run("avx2")
	}
}
