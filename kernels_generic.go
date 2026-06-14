//go:build !amd64 && !arm64 && !ppc64le && !s390x && !riscv64 && !loong64

package floats

// On architectures without a SIMD kernel, the reductions are the lane-blocked
// scalar reference itself — already a well-pipelined multi-accumulator loop.

func dot(a, b []float64) float64       { return dotLanes(a, b) }
func sum(a []float64) float64          { return sumLanes(a) }
func sumSqDiff(a, b []float64) float64 { return sumSqDiffLanes(a, b) }

func dot32(a, b []float32) float32       { return dotLanes32(a, b) }
func sum32(a []float32) float32          { return sumLanes32(a) }
func sumSqDiff32(a, b []float32) float32 { return sumSqDiffLanes32(a, b) }
