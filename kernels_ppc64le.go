//go:build ppc64le

package floats

import "golang.org/x/sys/cpu"

// ppc64le dispatch is split by element type, decided by real-POWER9 silicon
// (cfarm433, go1.26.4, 2026-06-27 — see kernels_ppc64le_gen.go):
//
//   - float64: VSX with EIGHT accumulators from vsxMin64 elements up. The earlier
//     single-accumulator kernel lost to the naive loop (0.82×, one xvmaddadp
//     chain of 6-7 cycles); with eight chains it wins on POWER9 (cfarm433, one
//     core): Dot 2.2× at n=1000, 2.4× at n=4096, 2.3× at n=1e5; Sum 2.8×, 3.0×,
//     3.0×, 3.0× at n=1e6; Distance 1.6×, 1.7×, 1.7×. Dot and Distance at n=1e6
//     are at memory bandwidth (1.05×). Below vsxMin64 the naive loop is faster.
//
//   - float32: VSX KEPT and now also enabled on POWER8. On POWER9 the f32 VSX
//     kernel WINS ~1.55–1.61× over the naive loop for n>=64. The kernel emits only
//     ISA-2.06 VSX ops (LXVW4X + xvmaddasp/xvaddsp/xvsubsp via WORD), all valid on
//     POWER8 — it was previously POWER9-gated out of caution, but it runs cleanly
//     on POWER8E (cfarm112) and there it WINS even bigger (the gc autovectorizer
//     is weaker on the POWER8 target, so the naive f32 loop is slow): measured
//     ~3.0× Dot at n=64, ~3.97× Dot / ~4.4× SumSqDiff / ~2.0× Sum at n=1024. So it
//     is gated behind hasVSX = POWER8+; pre-POWER8 (no AltiVec) falls back to the
//     naive f32 loop.
//
// All six kernels (three f32, three f64) are generated into kernels_ppc64le.s,
// each with eight accumulators; the f32 ones now win a further 1.9-2.4× (Dot
// 1.9× at n=1000, Sum 2.4×, n=1e6 Dot 2.0×, Sum 3.0×).
//
//go:noescape
func dot32VSX(a, b []float32) float32

//go:noescape
func sum32VSX(a []float32) float32

//go:noescape
func sumSqDiff32VSX(a, b []float32) float32

//go:noescape
func dot64VSX(a, b []float64) float64

//go:noescape
func sum64VSX(a []float64) float64

//go:noescape
func sumSqDiff64VSX(a, b []float64) float64

// vsxMin64 is the shortest float64 reduction sent to the VSX kernels: below it
// the kernel's setup (zeroing eight accumulators, the spill) costs 6-10 ns more
// than the plain loop (cfarm433, POWER9: n=3 9.5 vs 22 ns, n=8 15 vs 27 ns; at
// n=16 they tie, from n=32 the kernel wins 1.2x and more). The float32 kernels
// already tied the naive loop at every n, so they have no threshold. Under the
// threshold a four-accumulator Go loop (smallred.go) replaces the plain loop
// (cfarm29, POWER9, vs the wrapper with the plain loop: n=8 Dot 15.3 -> 15.0,
// Sum 13.8 -> 11.9 ns; n=12 Dot 21.2 -> 16.7, Sum 21.5 -> 13.4 ns; n=3 level).
const vsxMin64 = 16

// hasVSX gates the VSX kernels. They use only ISA-2.06 VSX ops (the ppc64le
// baseline is POWER8/ISA-2.07, which includes them), so they run on POWER8+;
// otherwise the naive f32 loop is used. It is a var (not a const) so the dispatch
// test can drive the fallback.
var hasVSX = cpu.PPC64.IsPOWER8

// ---- float64: naive autovectorizable loops (faster than VSX on POWER9) ----

func dot(a, b []float64) float64 {
	if len(a) >= vsxMin64 && hasVSX {
		return dot64VSX(a, b)
	}
	if len(a) >= 4 {
		return dotSmall(a, b)
	}
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sum(a []float64) float64 {
	if len(a) >= 4 {
		return sumMid(a)
	}
	var s float64
	for _, v := range a {
		s += v
	}
	return s
}

func sumMid(a []float64) float64 {
	if len(a) >= vsxMin64 && hasVSX {
		return sum64VSX(a)
	}
	return sumSmall(a)
}

func sumSqDiff(a, b []float64) float64 {
	if len(a) >= vsxMin64 && hasVSX {
		return sumSqDiff64VSX(a, b)
	}
	if len(a) >= 4 {
		return sumSqDiffSmall(a, b)
	}
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
