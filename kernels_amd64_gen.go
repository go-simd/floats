//go:build ignore

// Command gen produces kernels_amd64.s with go-asmgen: AVX2+FMA reduction
// kernels for the float32 and float64 dot product, sum and sum-of-squared-
// differences.
//
// Each kernel keeps one wide YMM accumulator (4×f64 or 8×f32), folds the bulk
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

func main() {
	f := emit.NewFile("amd64")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
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
