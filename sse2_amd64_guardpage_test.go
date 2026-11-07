//go:build amd64 && unix

package floats

import "testing"

// TestSSE2GuardPages runs the guard-page suites (nothing is read or written
// outside a slice) on the SSE2 kernels, with the AVX2+FMA gate off.
func TestSSE2GuardPages(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	t.Run("Binop", TestBinopGuardPages)
	t.Run("Elem", TestElemGuardPages)
}
