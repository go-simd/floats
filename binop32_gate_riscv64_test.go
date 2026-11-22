//go:build riscv64

package floats

// setBinop32Gate sets the CPU-feature gate of the float32 elementwise RVV kernels
// (hasRVV), returning the function that restores it and the previous value.
func setBinop32Gate(on bool) (restore func(), had bool) {
	saved := hasRVV
	hasRVV = on
	return func() { hasRVV = saved }, saved
}

// sqrt32Step: the emulated lane checks every 1021st bit pattern (about 4.2 million
// of them), not all 2^32.
const sqrt32Step = 1021

// runExhaustiveKernels runs the sampled check down the RVV kernel.
func runExhaustiveKernels(run func(label string)) { run("rvv") }
