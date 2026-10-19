//go:build ignore

// Command gen produces kernels_amd64.s with go-asmgen: AVX2+FMA reduction
// kernels for the float32 and float64 dot product, sum and sum-of-squared-
// differences, plus the float64 elementwise updates Axpy, Xpby, AxpyDot and
// Scale (see the section at the end of this file).
//
// Each reduction kernel keeps one wide YMM accumulator (4×f64 or 8×f32), folds the bulk
// of the slice into it with VFMADD231P{D,S} / VADDP{D,S} at the vector stride,
// then horizontally reduces the accumulator to a scalar and finishes the tail
// (< stride elements) with scalar FMA/add. The horizontal fold is
// VEXTRACTF128 + VADDP{D,S} (256→128) then per-element shuffles, so the kernel
// reduces in roughly the same lane-blocked order as the scalar reference; the
// differential tests pin the result to a tight ULP bound.
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
	if err := os.WriteFile("kernels_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_amd64.s")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX").
		Raw("VXORPD Y0, Y0, Y0"). // accumulator = 0
		Raw("XORQ DI, DI").       // i = 0
		Label("vloop").
		Raw("LEAQ 4(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VMOVUPD (AX)(DI*8), Y1").
		Raw("VMOVUPD (BX)(DI*8), Y2").
		Raw("VFMADD231PD Y2, Y1, Y0"). // acc += a*b
		Raw("ADDQ $4, DI").Raw("JMP vloop").
		Label("vtail")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").
		Raw("VXORPD Y0, Y0, Y0").
		Raw("XORQ DI, DI").
		Label("vloop").
		Raw("LEAQ 4(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VADDPD (AX)(DI*8), Y0, Y0").
		Raw("ADDQ $4, DI").Raw("JMP vloop").
		Label("vtail")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX").
		Raw("VXORPD Y0, Y0, Y0").
		Raw("XORQ DI, DI").
		Label("vloop").
		Raw("LEAQ 4(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VMOVUPD (AX)(DI*8), Y1").
		Raw("VSUBPD (BX)(DI*8), Y1, Y1"). // d = a-b
		Raw("VFMADD231PD Y1, Y1, Y0").    // acc += d*d
		Raw("ADDQ $4, DI").Raw("JMP vloop").
		Label("vtail")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX").
		Raw("VXORPS Y0, Y0, Y0").
		Raw("XORQ DI, DI").
		Label("vloop").
		Raw("LEAQ 8(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VMOVUPS (AX)(DI*4), Y1").
		Raw("VMOVUPS (BX)(DI*4), Y2").
		Raw("VFMADD231PS Y2, Y1, Y0").
		Raw("ADDQ $8, DI").Raw("JMP vloop").
		Label("vtail")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").
		Raw("VXORPS Y0, Y0, Y0").
		Raw("XORQ DI, DI").
		Label("vloop").
		Raw("LEAQ 8(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VADDPS (AX)(DI*4), Y0, Y0").
		Raw("ADDQ $8, DI").Raw("JMP vloop").
		Label("vtail")
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
	b.LoadArg("a_base", "AX").LoadArg("a_len", "CX").LoadArg("b_base", "BX").
		Raw("VXORPS Y0, Y0, Y0").
		Raw("XORQ DI, DI").
		Label("vloop").
		Raw("LEAQ 8(DI), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VMOVUPS (AX)(DI*4), Y1").
		Raw("VSUBPS (BX)(DI*4), Y1, Y1").
		Raw("VFMADD231PS Y1, Y1, Y0").
		Raw("ADDQ $8, DI").Raw("JMP vloop").
		Label("vtail")
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
