//go:build amd64

package floats

// sqrt32Step 1: every one of the 2^32 patterns.
const sqrt32Step = 1

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
