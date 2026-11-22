//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !riscv64 && !s390x

package floats

// No float32 kernel on this architecture: the portable loops, which the compiler
// already vectorises where it can.

func add32(dst, a, b []float32) { add32Ref(dst, a, b) }
func sub32(dst, a, b []float32) { sub32Ref(dst, a, b) }
func mul32(dst, a, b []float32) { mul32Ref(dst, a, b) }
func div32(dst, a, b []float32) { div32Ref(dst, a, b) }
func sqrt32(dst, src []float32) { sqrt32Ref(dst, src) }
