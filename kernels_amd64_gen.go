//go:build ignore

// Command gen produces kernels_amd64.s with go-asmgen: AVX2+FMA reduction
// kernels for the float32 and float64 dot product, sum and sum-of-squared-
// differences, plus the float64 elementwise updates Axpy, Xpby, AxpyDot and
// Scale (see the section at the end of this file).
//
// Each reduction kernel keeps redK (8) independent wide YMM accumulators (4×f64
// or 8×f32 each; see redLoops), folds the bulk of the slice into them with
// VFMADD231P{D,S} / VADDP{D,S}, sums them in a tree, then horizontally reduces
// to a scalar and finishes the tail (< stride elements) with scalar FMA/add. The
// summation order is therefore not the lane-blocked order of the scalar
// reference: reduce_bound_test.go asserts each result against a math/big oracle
// with the gamma_n * sum|terms| bound.
//
// Operand order: Plan 9 AVX three-operand form is `OP src2, src1, dst`, and the
// 231 FMA computes dst += src1*src2 — i.e. `VFMADD231PD b, a, acc` is
// acc += a*b. Dispatch gates on cpu.X86.HasAVX2 && HasFMA; the SSE2 baseline
// path is the scalar reference.
//
// Run: go run kernels_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func dotSig(elem abi.Type) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", elem)},
	)
}

func sumSig(elem abi.Type) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", elem)},
	)
}

// axpySig is func(<scalar> float64, x, y []float64): the scalar comes first, so
// it sits at 0(FP) and the slice headers follow at 8 and 32.
func axpySig(scalar string) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar(scalar, abi.Float64), abi.Slice("x"), abi.Slice("y")},
		nil,
	)
}

// axpyDotSig is func(alpha float64, x, y, w []float64) float64.
func axpyDotSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("alpha", abi.Float64), abi.Slice("x"), abi.Slice("y"), abi.Slice("w")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)},
	)
}

// scaleSig is func(alpha float64, x []float64).
func scaleSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("alpha", abi.Float64), abi.Slice("x")},
		nil,
	)
}

func main() {
	f := emit.NewFile("amd64")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	f.Add(axpyF64().Func())
	f.Add(xpbyF64().Func())
	f.Add(axpyDotF64().Func())
	f.Add(scaleF64().Func())
	f.Add(mulF64().Func())
	f.Add(axpby3F64().Func())
	f.Add(clampF64().Func())
	f.Add(maxAbsF64().Func())
	if err := os.WriteFile("kernels_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_amd64.s")
}

// ---- multi-accumulator reduction skeleton ----
//
// A reduction folds each product (or element) into an accumulator with a
// loop-carried dependency: one FMA/ADD per iteration is bound by the
// instruction's LATENCY (4 cycles for VFMADD on Zen 3/4 and Intel, 2 ports),
// not by its throughput, so a single accumulator runs at 1/4..1/8 of the load
// bandwidth on cache-resident data. redK independent YMM accumulators
// (Y0..Y{redK-1}) are filled K*lanes elements per pass (the ×K loop), summed in
// a balanced tree (halving), and the sub-K*lanes remainder is folded into Y0 one
// vector at a time (the vloop, which is also the whole path for n < K*lanes, so
// short vectors cost what they cost before), before the horizontal reduce and
// the scalar tail the caller emits after Label("vtail").
//
// The order of summation changes with K; the accuracy contract is the standard
// recursive-summation bound gamma_n * sum|terms| (see the oracle tests).
const redK = 8

// redLoops emits the zeroing, the ×redK loop, the accumulator tree, and the
// single-vector loop, ending at Label("vtail") with the total in Y0. body emits
// the update of accumulator acc from the elements at byte offset off of the
// vector index, using Y(8+) temp t; m(base, off) is the addressing helper.
func redLoops(b *amd64.Builder, f32 bool, body func(acc string, off int, t string, m func(base string, off int) string)) {
	lanes, esz, xor, add := 4, 8, "VXORPD", "VADDPD"
	if f32 {
		lanes, esz, xor, add = 8, 4, "VXORPS", "VADDPS"
	}
	m := func(base string, off int) string {
		if off == 0 {
			return fmt.Sprintf("(%s)(DI*%d)", base, esz)
		}
		return fmt.Sprintf("%d(%s)(DI*%d)", off, base, esz)
	}
	// Short vectors skip the multi-accumulator machinery (zeroing, and the
	// log2(K)-deep add tree, which alone costs ~9 cycles of latency): fewer than
	// one vector goes straight to the scalar tail, fewer than K vectors to the
	// single-accumulator loop.
	b.Raw("XORQ DI, DI").
		Raw("CMPQ CX, $%d", lanes).Raw("JGE notiny").
		Raw("%s X0, X0, X0", xor).Raw("JMP sloop").
		Label("notiny").
		Raw("CMPQ CX, $%d", redK*lanes).Raw("JLT medium")
	for k := 0; k < redK; k++ {
		b.Raw("%s Y%d, Y%d, Y%d", xor, k, k, k)
	}
	b.Label("uloop").
		Raw("LEAQ %d(DI), R8", redK*lanes).Raw("CMPQ R8, CX").Raw("JGT cloop")
	for k := 0; k < redK; k++ {
		body(fmt.Sprintf("Y%d", k), 32*k, fmt.Sprintf("Y%d", 8+k), m)
	}
	b.Raw("ADDQ $%d, DI", redK*lanes).Raw("JMP uloop").
		Label("cloop")
	for h := redK / 2; h >= 1; h /= 2 {
		for k := 0; k < h; k++ {
			b.Raw("%s Y%d, Y%d, Y%d", add, k+h, k, k)
		}
	}
	b.Raw("JMP vloop").
		Label("medium").
		Raw("%s Y0, Y0, Y0", xor).
		Label("vloop").
		Raw("LEAQ %d(DI), R8", lanes).Raw("CMPQ R8, CX").Raw("JGT vtail")
	body("Y0", 0, "Y8", m)
	b.Raw("ADDQ $%d, DI", lanes).Raw("JMP vloop").
		Label("vtail")
}

// ---- float64 (4 lanes/YMM, 8-byte elements) ----

// reduceF64 folds Y0 (4×f64) down to the scalar in X0 low lane.
func reduceF64(b *amd64.Builder) {
	b.Raw("VEXTRACTF128 $1, Y0, X1"). // high 128
						Raw("VADDPD X1, X0, X0").  // 2 lanes
						Raw("VHADDPD X0, X0, X0"). // low lane = lane0+lane1
						Raw("VZEROUPPER")
}

func dotF64() *amd64.Builder {
	b := amd64.NewFunc("dotAVX2", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX")
	redLoops(b, false, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VMOVUPD %s, %s", m("AX", off), t).
			Raw("VFMADD231PD %s, %s, %s", m("BX", off), t, acc) // acc += a*b
	})
	reduceF64(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VMOVSD (AX)(DI*8), X1").
		Raw("VMOVSD (BX)(DI*8), X2").
		Raw("VFMADD231SD X2, X1, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

func sumF64() *amd64.Builder {
	b := amd64.NewFunc("sumAVX2", sumSig(abi.Float64), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX")
	redLoops(b, false, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VADDPD %s, %s, %s", m("AX", off), acc, acc)
	})
	reduceF64(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VADDSD (AX)(DI*8), X0, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

func ssdF64() *amd64.Builder {
	b := amd64.NewFunc("sumSqDiffAVX2", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX")
	redLoops(b, false, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VMOVUPD %s, %s", m("AX", off), t).
			Raw("VSUBPD %s, %s, %s", m("BX", off), t, t). // d = a-b
			Raw("VFMADD231PD %s, %s, %s", t, t, acc)      // acc += d*d
	})
	reduceF64(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VMOVSD (AX)(DI*8), X1").
		Raw("VSUBSD (BX)(DI*8), X1, X1").
		Raw("VFMADD231SD X1, X1, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

// ---- float32 (8 lanes/YMM, 4-byte elements) ----

// reduceF32 folds Y0 (8×f32) down to the scalar in X0 low lane.
func reduceF32(b *amd64.Builder) {
	b.Raw("VEXTRACTF128 $1, Y0, X1"). // high 128 (4 lanes)
						Raw("VADDPS X1, X0, X0").  // 4 lanes
						Raw("VHADDPS X0, X0, X0"). // -> 2 partial sums
						Raw("VHADDPS X0, X0, X0"). // low lane = total
						Raw("VZEROUPPER")
}

func dotF32() *amd64.Builder {
	b := amd64.NewFunc("dot32AVX2", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX")
	redLoops(b, true, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VMOVUPS %s, %s", m("AX", off), t).
			Raw("VFMADD231PS %s, %s, %s", m("BX", off), t, acc)
	})
	reduceF32(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VMOVSS (AX)(DI*4), X1").
		Raw("VMOVSS (BX)(DI*4), X2").
		Raw("VFMADD231SS X2, X1, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

func sumF32() *amd64.Builder {
	b := amd64.NewFunc("sum32AVX2", sumSig(abi.Float32), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX")
	redLoops(b, true, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VADDPS %s, %s, %s", m("AX", off), acc, acc)
	})
	reduceF32(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VADDSS (AX)(DI*4), X0, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

func ssdF32() *amd64.Builder {
	b := amd64.NewFunc("sumSqDiff32AVX2", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX")
	redLoops(b, true, func(acc string, off int, t string, m func(string, int) string) {
		b.Raw("VMOVUPS %s, %s", m("AX", off), t).
			Raw("VSUBPS %s, %s, %s", m("BX", off), t, t).
			Raw("VFMADD231PS %s, %s, %s", t, t, acc)
	})
	reduceF32(b)
	b.Label("sloop").
		Raw("CMPQ DI, CX").Raw("JGE done").
		Raw("VMOVSS (AX)(DI*4), X1").
		Raw("VSUBSS (BX)(DI*4), X1, X1").
		Raw("VFMADD231SS X1, X1, X0").
		Raw("ADDQ $1, DI").Raw("JMP sloop").
		Label("done").StoreRet("X0", "ret").Ret()
	return b
}

// ---- float64 elementwise update kernels (BLAS level 1) ----
//
// Register plan: Y0 = the broadcast scalar (alpha or beta; X0 keeps it in the
// low lane for the scalar tail), SI = x, DI = y, DX = w, CX = n, AX = i.
//
// Every element is ONE fused multiply-add, so each result is bit-identical to
// math.FMA(alpha, x[i], y[i]) (or math.FMA(beta, y[i], x[i]) for Xpby): the
// packed VFMADD and the scalar tail VFMADD round exactly once, like math.FMA.
// The bulk loop is unrolled ×4 (16 elements per iteration) so four independent
// load/FMA/store chains are in flight; a single-vector loop and a scalar tail
// finish the remainder. Every access is at an index < n: the unrolled loop runs
// only while i+16 <= n, the vector loop while i+4 <= n, and the tail is scalar,
// so no kernel reads or writes past the end of a slice (lengths 0..3 never
// touch a vector load).
//
// The 213 form: Plan 9 `VFMADD213PD src2, src1, dst` is dst = src1*dst + src2.

// elemLoop emits the shared ×4 / ×1 / scalar loop skeleton. vec(off, reg)
// emits the per-vector body for the YMM register reg at byte offset off from
// the element index; sca() emits the scalar tail body for X1.
func elemLoop(b *amd64.Builder, vec func(off int, reg string), sca func()) {
	regs := []string{"Y1", "Y2", "Y3", "Y4"}
	b.Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 16(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	for k, r := range regs {
		vec(32*k, r)
	}
	b.Raw("ADDQ $16, AX").Raw("JMP uloop").
		Label("vloop").
		Raw("LEAQ 4(AX), R8").Raw("CMPQ R8, CX").Raw("JGT sloop")
	vec(0, "Y1")
	b.Raw("ADDQ $4, AX").Raw("JMP vloop").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done")
	sca()
	b.Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done")
}

func mem(base string, off int) string {
	if off == 0 {
		return fmt.Sprintf("(%s)(AX*8)", base)
	}
	return fmt.Sprintf("%d(%s)(AX*8)", off, base)
}

// axpyF64: y[i] = fma(alpha, x[i], y[i]).
func axpyF64() *amd64.Builder {
	b := amd64.NewFunc("axpyAVX2", axpySig("alpha"), 0)
	b.LoadArg("alpha", "X0").LoadArg("x_base", "SI").LoadArg("x_len", "CX").LoadArg("y_base", "DI").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("DI", off), r).
			Raw("VFMADD231PD %s, Y0, %s", mem("SI", off), r). // y += alpha*x
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (DI)(AX*8), X1").
			Raw("VFMADD231SD (SI)(AX*8), X0, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// xpbyF64: y[i] = fma(beta, y[i], x[i]).
func xpbyF64() *amd64.Builder {
	b := amd64.NewFunc("xpbyAVX2", axpySig("beta"), 0)
	b.LoadArg("beta", "X0").LoadArg("x_base", "SI").LoadArg("x_len", "CX").LoadArg("y_base", "DI").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("DI", off), r).
			Raw("VFMADD213PD %s, Y0, %s", mem("SI", off), r). // y = beta*y + x
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (DI)(AX*8), X1").
			Raw("VFMADD213SD (SI)(AX*8), X0, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// axpyDotF64: y[i] = fma(alpha, x[i], y[i]) and returns Σ y[i]*w[i] over the
// updated y, in one pass. The dot keeps four YMM accumulators (Y5..Y8, one per
// unrolled vector) that are summed pairwise, folded 4→1 like the Dot kernel,
// and then extended by the scalar tail.
func axpyDotF64() *amd64.Builder {
	b := amd64.NewFunc("axpyDotAVX2", axpyDotSig(), 0)
	b.LoadArg("alpha", "X0").LoadArg("x_base", "SI").LoadArg("x_len", "CX").
		LoadArg("y_base", "DI").LoadArg("w_base", "DX").
		Raw("VBROADCASTSD X0, Y0").
		Raw("VXORPD Y5, Y5, Y5").Raw("VXORPD Y6, Y6, Y6").
		Raw("VXORPD Y7, Y7, Y7").Raw("VXORPD Y8, Y8, Y8").
		Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 16(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	accs := []string{"Y5", "Y6", "Y7", "Y8"}
	for k, r := range []string{"Y1", "Y2", "Y3", "Y4"} {
		off := 32 * k
		b.Raw("VMOVUPD %s, %s", mem("DI", off), r).
			Raw("VFMADD231PD %s, Y0, %s", mem("SI", off), r).
			Raw("VMOVUPD %s, %s", r, mem("DI", off)).
			Raw("VFMADD231PD %s, %s, %s", mem("DX", off), r, accs[k]) // acc += y*w
	}
	b.Raw("ADDQ $16, AX").Raw("JMP uloop").
		Label("vloop").
		Raw("VADDPD Y6, Y5, Y5").Raw("VADDPD Y8, Y7, Y7").Raw("VADDPD Y7, Y5, Y5").
		Label("vloop1").
		Raw("LEAQ 4(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VMOVUPD (DI)(AX*8), Y1").
		Raw("VFMADD231PD (SI)(AX*8), Y0, Y1").
		Raw("VMOVUPD Y1, (DI)(AX*8)").
		Raw("VFMADD231PD (DX)(AX*8), Y1, Y5").
		Raw("ADDQ $4, AX").Raw("JMP vloop1").
		Label("vtail").
		Raw("VEXTRACTF128 $1, Y5, X6").
		Raw("VADDPD X6, X5, X5").
		Raw("VHADDPD X5, X5, X5").
		Raw("VZEROUPPER").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done").
		Raw("VMOVSD (DI)(AX*8), X1").
		Raw("VFMADD231SD (SI)(AX*8), X0, X1").
		Raw("VMOVSD X1, (DI)(AX*8)").
		Raw("VFMADD231SD (DX)(AX*8), X1, X5").
		Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done").StoreRet("X5", "ret").Ret()
	return b
}

// scaleF64: x[i] *= alpha (one rounding, like the scalar multiply).
func scaleF64() *amd64.Builder {
	b := amd64.NewFunc("scaleAVX2", scaleSig(), 0)
	b.LoadArg("alpha", "X0").LoadArg("x_base", "DI").LoadArg("x_len", "CX").
		Raw("VBROADCASTSD X0, Y0")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMULPD %s, Y0, %s", mem("DI", off), r).
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMULSD (DI)(AX*8), X0, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// ---- Mul, Axpby3, Clamp, MaxAbs ----
//
// Same skeleton as above (elemLoop: ×4 unrolled, then 4-wide, then scalar, every
// access below n). Registers: DI = dst, SI = first input, DX = second, R9 =
// third; each 4-vector block is loaded, computed and stored before the next, so
// dst may be the same slice as any input.

// mulSig is func(dst, a, b []float64).
func mulSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
}

// mulF64: dst[i] = a[i]*b[i].
func mulF64() *amd64.Builder {
	b := amd64.NewFunc("mulAVX2", mulSig(), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("SI", off), r).
			Raw("VMULPD %s, %s, %s", mem("DX", off), r, r).
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (SI)(AX*8), X1").
			Raw("VMULSD (DX)(AX*8), X1, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// axpby3Sig is func(dst []float64, alpha float64, a []float64, beta float64,
// b []float64, gamma float64, c []float64).
func axpby3Sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Scalar("alpha", abi.Float64), abi.Slice("a"),
			abi.Scalar("beta", abi.Float64), abi.Slice("b"),
			abi.Scalar("gamma", abi.Float64), abi.Slice("c")}, nil)
}

// axpby3F64: dst[i] = fma(gamma, c[i], fma(beta, b[i], alpha*a[i])). Y0/Y10/Y11 =
// broadcast alpha/beta/gamma (X0/X10/X11 keep them for the scalar tail).
func axpby3F64() *amd64.Builder {
	b := amd64.NewFunc("axpby3AVX2", axpby3Sig(), 0)
	b.LoadArg("alpha", "X0").LoadArg("beta", "X10").LoadArg("gamma", "X11").
		LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").
		LoadArg("b_base", "DX").LoadArg("c_base", "R9").
		Raw("VBROADCASTSD X0, Y0").Raw("VBROADCASTSD X10, Y10").Raw("VBROADCASTSD X11, Y11")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMULPD %s, Y0, %s", mem("SI", off), r). // r = alpha*a
								Raw("VFMADD231PD %s, Y10, %s", mem("DX", off), r). // r += beta*b
								Raw("VFMADD231PD %s, Y11, %s", mem("R9", off), r). // r += gamma*c
								Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMULSD (SI)(AX*8), X0, X1").
			Raw("VFMADD231SD (DX)(AX*8), X10, X1").
			Raw("VFMADD231SD (R9)(AX*8), X11, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// clampSig is func(dst, v, lo, hi []float64).
func clampSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("v"), abi.Slice("lo"), abi.Slice("hi")}, nil)
}

// clampF64: dst[i] = (lo>v ? lo : v), then (hi<r ? hi : r).
//
// VMAXPD/VMINPD return their SECOND source (the memory/last Plan 9 operand) when
// the operands are equal or either is a NaN, and the compare is the strict one
// the reference uses: MAX(src1, src2) = src1 > src2 ? src1 : src2, so
// `VMAXPD v, lo, r` is lo > v ? lo : v — v wins ties (signed zeros) and a NaN
// v, and a NaN lo is ignored. `VMINPD r, hi, r` is hi < r ? hi : r likewise. A
// NaN v therefore passes through both steps with its own bits.
func clampF64() *amd64.Builder {
	b := amd64.NewFunc("clampAVX2", clampSig(), 0)
	b.LoadArg("dst_base", "DI").LoadArg("v_base", "SI").LoadArg("v_len", "CX").
		LoadArg("lo_base", "DX").LoadArg("hi_base", "R9")
	elemLoop(b, func(off int, r string) {
		b.Raw("VMOVUPD %s, %s", mem("DX", off), r).
			Raw("VMAXPD %s, %s, %s", mem("SI", off), r, r).
			Raw("VMOVUPD %s, Y5", mem("R9", off)).
			Raw("VMINPD %s, Y5, %s", r, r).
			Raw("VMOVUPD %s, %s", r, mem("DI", off))
	}, func() {
		b.Raw("VMOVSD (DX)(AX*8), X1").
			Raw("VMAXSD (SI)(AX*8), X1, X1").
			Raw("VMOVSD (R9)(AX*8), X5").
			Raw("VMINSD X1, X5, X1").
			Raw("VMOVSD X1, (DI)(AX*8)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// maxAbsSig is func(v []float64) float64.
func maxAbsSig() abi.Signature { return sumSig(abi.Float64) }

// maxAbsF64 returns max |v[i]|, or a NaN if any element is a NaN.
//
// |x| is x AND 0x7fff…: the mask is built in Y15 (all ones, shifted right by one)
// so no data section is needed. VMAXPD is not NaN-sticky (it returns its second
// source when unordered), so NaNs are tracked on the side: VCMPPD $3 (unordered)
// of |x| with itself sets a lane, ORed into Y4. Four max accumulators (Y0..Y3)
// hide the VMAXPD latency; they and the NaN lanes are folded before the scalar
// tail, which runs on the low lane (a VEX.128 instruction zeroes the upper half,
// so nothing vector-wide may be live then). The result is OR-ed with the NaN mask:
// all ones is a NaN.
func maxAbsF64() *amd64.Builder {
	b := amd64.NewFunc("maxAbsAVX2", maxAbsSig(), 0)
	b.LoadArg("a_base", "SI").LoadArg("a_len", "CX").
		Raw("VPCMPEQD Y15, Y15, Y15").Raw("VPSRLQ $1, Y15, Y15").
		Raw("VXORPD Y0, Y0, Y0").Raw("VXORPD Y1, Y1, Y1").
		Raw("VXORPD Y2, Y2, Y2").Raw("VXORPD Y3, Y3, Y3").Raw("VXORPD Y4, Y4, Y4").
		Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 16(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	for k := 0; k < 4; k++ {
		x := fmt.Sprintf("Y%d", 5+k)
		m := fmt.Sprintf("Y%d", 9+k)
		b.Raw("VANDPD %s, Y15, %s", mem("SI", 32*k), x).
			Raw("VMAXPD %s, Y%d, Y%d", x, k, k).
			Raw("VCMPPD $3, %s, %s, %s", x, x, m).
			Raw("VORPD %s, Y4, Y4", m)
	}
	b.Raw("ADDQ $16, AX").Raw("JMP uloop").
		Label("vloop").
		Raw("VMAXPD Y1, Y0, Y0").Raw("VMAXPD Y3, Y2, Y2").Raw("VMAXPD Y2, Y0, Y0").
		Label("vloop1").
		Raw("LEAQ 4(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VANDPD (SI)(AX*8), Y15, Y5").
		Raw("VMAXPD Y5, Y0, Y0").
		Raw("VCMPPD $3, Y5, Y5, Y9").
		Raw("VORPD Y9, Y4, Y4").
		Raw("ADDQ $4, AX").Raw("JMP vloop1").
		Label("vtail").
		Raw("VEXTRACTF128 $1, Y0, X5").Raw("VMAXPD X5, X0, X0").
		Raw("VUNPCKHPD X0, X0, X5").Raw("VMAXSD X5, X0, X0").
		Raw("VEXTRACTF128 $1, Y4, X6").Raw("VORPD X6, X4, X4").
		Raw("VUNPCKHPD X4, X4, X6").Raw("VORPD X6, X4, X4").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done").
		Raw("VMOVSD (SI)(AX*8), X5").
		Raw("VANDPD X15, X5, X5").
		Raw("VMAXSD X5, X0, X0").
		Raw("VCMPSD $3, X5, X5, X9").
		Raw("VORPD X9, X4, X4").
		Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done").
		Raw("VORPD X4, X0, X0").
		Raw("VZEROUPPER").
		StoreRet("X0", "ret").Ret()
	return b
}
