//go:build s390x

package floats

// setBinop32Gate sets the CPU-feature gate of the float32 elementwise VXE kernels
// (hasVXE), returning the function that restores it and the previous value.
func setBinop32Gate(on bool) (restore func(), had bool) {
	saved := hasVXE
	hasVXE = on
	return func() { hasVXE = saved }, saved
}

// sqrt32Step: the emulated lane checks every 1021st bit pattern (about 4.2 million
// of them), not all 2^32.
const sqrt32Step = 1021

// runExhaustiveKernels runs the sampled check down the VXE kernel.
func runExhaustiveKernels(run func(label string)) { run("vxe") }
