//go:build arm64

package floats

// runExhaustiveKernels runs the exhaustive check down the NEON kernel.
func runExhaustiveKernels(run func(label string)) { run("neon") }
