//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !riscv64 && !s390x

package floats

// No float32 Max/Min kernel on this architecture: the four-chain loops.

func maxF32(a []float32) float32 { return maxRef32(a) }
func minF32(a []float32) float32 { return minRef32(a) }
