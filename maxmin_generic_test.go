//go:build !amd64 && !arm64 && !s390x && !ppc64le && !loong64 && !riscv64

package floats

// No kernel runs on this architecture: Maximum and Minimum are the builtin loop.
func init() { pairwiseFallback = true }
