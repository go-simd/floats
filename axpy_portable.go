//go:build !amd64 && !arm64

package floats

// The elementwise updates have assembly kernels on amd64 and arm64 only, the
// two architectures where they were measured against the scalar loop.
//
// On ppc64le, s390x, riscv64 and loong64 they run the portable math.FMA loops
// of axpy.go, which the compiler lowers to one hardware FMA per element. The
// reductions in this package went the same way on those targets: their vector
// kernels were dropped or kept only after measurement on real silicon (POWER9,
// z15, SpacemiT X60, LA464), several of them losing to the compiled loop, and
// none of that hardware is reachable from here — emulated timings say nothing
// about it. An elementwise update has no loop-carried dependency and is bound
// by memory traffic at large n, so the scalar FMA loop is the conservative
// choice until a kernel can be measured on the hardware.
//
// The updated slices are bit-identical to the amd64 and arm64 kernels' either
// way; only AxpyDot's returned sum may differ in the last bits (lane order).

func axpy(alpha float64, x, y []float64)               { axpyRef(alpha, x, y) }
func xpby(beta float64, x, y []float64)                { xpbyRef(beta, x, y) }
func axpyDot(alpha float64, x, y, w []float64) float64 { return axpyDotRef(alpha, x, y, w) }
func scale(alpha float64, x []float64)                 { scaleRef(alpha, x) }
