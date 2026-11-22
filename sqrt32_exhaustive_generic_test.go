//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !riscv64 && !s390x

package floats

// sqrt32Step: no kernel on this architecture, so a sample (every 1021st bit
// pattern) of the portable loop is enough.
const sqrt32Step = 1021

// runExhaustiveKernels runs the sampled check down the portable loop.
func runExhaustiveKernels(run func(label string)) { run("portable") }
