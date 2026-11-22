//go:build arm64

package floats

// sqrt32Step 1: every one of the 2^32 patterns.
const sqrt32Step = 1

// runExhaustiveKernels runs the exhaustive check down the NEON kernel.
func runExhaustiveKernels(run func(label string)) { run("neon") }
