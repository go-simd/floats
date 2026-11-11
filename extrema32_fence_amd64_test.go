//go:build amd64 && (linux || darwin)

package floats

import "testing"

// TestFloat32ExtremaFencedSSE2 is the fence test with the AVX2 gate off, so the
// SSE2 kernels are fenced at every length too (the default run fences AVX2 where
// the CPU has it).
func TestFloat32ExtremaFencedSSE2(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestFloat32ExtremaFenced(t)
}
