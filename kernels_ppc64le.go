//go:build ppc64le

package floats

import "golang.org/x/sys/cpu"

// ppc64le dispatch is split by element type, decided by real-POWER9 silicon
// (cfarm433, go1.26.4, 2026-06-27 — see kernels_ppc64le_gen.go):
//
//   - float64: NO VSX. The gc compiler already autovectorizes the plain Go
//     reduction loop, and on POWER9 that autovectorized scalar BEATS the f64 VSX
//     kernel — measured Dot throughput VSX vs naive 0.91× (n=8), 0.92× (n=64),
//     0.82× (n=512), 0.82× (n=4096); Sum and SumSqDiff the same ~0.82×. Per the
//     dispatch principle (never run a SIMD kernel where it loses to the
//     scalar/autovectorized path) the f64 reductions route to the naive loops
//     below, which the compiler vectorizes into the faster code. We deliberately
//     do NOT use the lane-blocked reference (reference.go) here: its multi-lane
//     fold defeats autovectorization and is ~2× slower than both naive and VSX.
//
//   - float32: VSX KEPT. On the same POWER9 the f32 VSX kernel WINS ~1.55–1.61×
//     over the naive loop for n>=64 (it only trails at the tiny n=8). The kernel
//     emits ISA-3.0 (POWER9) VSX instructions that SIGILL on POWER8, so it is
//     gated behind hasVSX (cpu.PPC64.IsPOWER9); pre-POWER9 falls back to the
//     naive f32 loop.
//
// Only the three f32 kernels are generated into kernels_ppc64le.s.
func dot32VSX(a, b []float32) float32
func sum32VSX(a []float32) float32
func sumSqDiff32VSX(a, b []float32) float32

// hasVSX gates the f32 VSX kernels. They use ISA-3.0 (POWER9) instructions
// illegal on POWER8, so they run only on POWER9+; otherwise the naive f32 loop
// is used. It is a var (not a const) so the dispatch test can drive the
// fallback.
var hasVSX = cpu.PPC64.IsPOWER9

// ---- float64: naive autovectorizable loops (faster than VSX on POWER9) ----

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sum(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v
	}
	return s
}

func sumSqDiff(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// ---- float32: VSX kernel on POWER9, naive autovectorizable fallback ----

func dot32(a, b []float32) float32 {
	if hasVSX {
		return dot32VSX(a, b)
	}
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sum32(a []float32) float32 {
	if hasVSX {
		return sum32VSX(a)
	}
	var s float32
	for _, v := range a {
		s += v
	}
	return s
}

func sumSqDiff32(a, b []float32) float32 {
	if hasVSX {
		return sumSqDiff32VSX(a, b)
	}
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}
