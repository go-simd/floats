//go:build ignore

// Command gen produces kernels_loong64.s with go-asmgen: LSX (128-bit vector)
// FMA reduction kernels for the float32/float64 dot product, sum and
// sum-of-squared-differences.
//
// The released Go assembler has the LSX vector load/store (VMOVQ) and the
// scalar FP ops (MOVD/MOVF, ADDD/ADDF, SUBD/SUBF, FMADDD/FMADDF), but NOT the
// LSX floating-point arithmetic (vfmadd/vfadd/vfsub/vxor), so those are emitted
// as their fixed LoongArch encodings via WORD. The encodings come from GNU as
// and are commented inline; the register layout is pinned (acc=V2=$vr2, a=V0,
// b=V1, diff=V3) so the constants are stable.
//
//	WORD $0x71270842  vxor.v   $vr2,$vr2,$vr2       acc = 0
//	WORD $0x09210402  vfmadd.d $vr2,$vr0,$vr1,$vr2  acc += a*b   (f64)
//	WORD $0x71310802  vfadd.d  $vr2,$vr0,$vr2       acc += a     (f64)
//	WORD $0x71330403  vfsub.d  $vr3,$vr0,$vr1       d = a-b      (f64)
//	WORD $0x09210c62  vfmadd.d $vr2,$vr3,$vr3,$vr2  acc += d*d   (f64)
//	WORD $0x09110402  vfmadd.s $vr2,$vr0,$vr1,$vr2  acc += a*b   (f32)
//	WORD $0x71308802  vfadd.s  $vr2,$vr0,$vr2       acc += a     (f32)
//	WORD $0x71328403  vfsub.s  $vr3,$vr0,$vr1       d = a-b      (f32)
//	WORD $0x09110c62  vfmadd.s $vr2,$vr3,$vr3,$vr2  acc += d*d   (f32)
//
// Each kernel folds the bulk into V2 (2×f64 / 4×f32) at the vector stride, spills
// it to a 16-byte stack scratch with VMOVQ, sums the lanes in scalar FPRs and
// finishes the < stride tail. Lane numbering is irrelevant: every op is an
// element-wise reduction with a commutative horizontal sum. LSX is the standard
// LA464 baseline, so there is no runtime feature gate.
//
// Scalar operand orders (verified on qemu): ADDD/SUBD/MULD `A, B, D` → D = A op
// B (so SUBD b, a → a-b); FMADDD/FMADDF `A, B, C, D` → D = A + B*C (the FIRST
// operand is the addend).
//
// Run: go run kernels_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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
	f := emit.NewFile("loong64")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	if err := os.WriteFile("kernels_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_loong64.s")
}

// foldF64 spills V2 (2×f64) to stack scratch at SP+32 and sums the two lanes.
func foldF64(b *loong64.Builder) {
	b.Raw("MOVV $32, R11").Raw("ADDV R3, R11, R11").
		Raw("VMOVQ V2, (R11)").
		Raw("MOVD (R11), F0").
		Raw("MOVD 8(R11), F1").
		Raw("ADDD F0, F1, F0") // F0 = F0 + F1
}

// foldF32 spills V2 (4×f32) and sums the four lanes.
func foldF32(b *loong64.Builder) {
	b.Raw("MOVV $32, R11").Raw("ADDV R3, R11, R11").
		Raw("VMOVQ V2, (R11)").
		Raw("MOVF (R11), F0").
		Raw("MOVF 4(R11), F1").
		Raw("MOVF 8(R11), F2").
		Raw("MOVF 12(R11), F3").
		Raw("ADDF F0, F1, F0").
		Raw("ADDF F0, F2, F0").
		Raw("ADDF F0, F3, F0")
}

// Register plan: R4 a_base, R5 a_len, R6 b_base (LoadArg); R7 i; R8 byte off;
// R9 i+stride; R3 is SP (frame base for the scratch spill).

// ---- float64 (V2 = 2 lanes, stride 2) ----

func dotF64() *loong64.Builder {
	b := loong64.NewFunc("dotLSX", dotSig(abi.Float64), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("WORD $0x71270842"). // vxor.v acc
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $2, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("ADDV R6, R8, R11").Raw("VMOVQ (R11), V1").
		Raw("WORD $0x09210402"). // vfmadd.d acc += a*b
		Raw("ADDV $2, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVD (R10), F1").
		Raw("ADDV R6, R8, R11").Raw("MOVD (R11), F2").
		Raw("FMADDD F0, F1, F2, F0"). // F0 = a*b + F0
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF64() *loong64.Builder {
	b := loong64.NewFunc("sumLSX", sumSig(abi.Float64), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").
		Raw("WORD $0x71270842").
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $2, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("WORD $0x71310802"). // vfadd.d acc += a
		Raw("ADDV $2, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVD (R10), F1").
		Raw("ADDD F0, F1, F0").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF64() *loong64.Builder {
	b := loong64.NewFunc("sumSqDiffLSX", dotSig(abi.Float64), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("WORD $0x71270842").
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $2, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("ADDV R6, R8, R11").Raw("VMOVQ (R11), V1").
		Raw("WORD $0x71330403"). // vfsub.d d = a-b
		Raw("WORD $0x09210c62"). // vfmadd.d acc += d*d
		Raw("ADDV $2, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $3, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVD (R10), F1").
		Raw("ADDV R6, R8, R11").Raw("MOVD (R11), F2").
		Raw("SUBD F2, F1, F1"). // F1 = a - b
		Raw("FMADDD F0, F1, F1, F0").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// ---- float32 (V2 = 4 lanes, stride 4) ----

func dotF32() *loong64.Builder {
	b := loong64.NewFunc("dot32LSX", dotSig(abi.Float32), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("WORD $0x71270842").
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $4, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("ADDV R6, R8, R11").Raw("VMOVQ (R11), V1").
		Raw("WORD $0x09110402"). // vfmadd.s acc += a*b
		Raw("ADDV $4, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVF (R10), F1").
		Raw("ADDV R6, R8, R11").Raw("MOVF (R11), F2").
		Raw("FMADDF F0, F1, F2, F0").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF32() *loong64.Builder {
	b := loong64.NewFunc("sum32LSX", sumSig(abi.Float32), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").
		Raw("WORD $0x71270842").
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $4, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("WORD $0x71308802"). // vfadd.s acc += a
		Raw("ADDV $4, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVF (R10), F1").
		Raw("ADDF F0, F1, F0").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF32() *loong64.Builder {
	b := loong64.NewFunc("sumSqDiff32LSX", dotSig(abi.Float32), 48)
	b.LoadArg("a_base", "R4").LoadArg("a_len", "R5").LoadArg("b_base", "R6").
		Raw("WORD $0x71270842").
		Raw("MOVV $0, R7").
		Label("vloop").
		Raw("ADDV $4, R7, R9").Raw("BLT R5, R9, vtail").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("VMOVQ (R10), V0").
		Raw("ADDV R6, R8, R11").Raw("VMOVQ (R11), V1").
		Raw("WORD $0x71328403"). // vfsub.s d = a-b
		Raw("WORD $0x09110c62"). // vfmadd.s acc += d*d
		Raw("ADDV $4, R7, R7").Raw("JMP vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("BGE R7, R5, done").
		Raw("SLLV $2, R7, R8").
		Raw("ADDV R4, R8, R10").Raw("MOVF (R10), F1").
		Raw("ADDV R6, R8, R11").Raw("MOVF (R11), F2").
		Raw("SUBF F2, F1, F1").
		Raw("FMADDF F0, F1, F1, F0").
		Raw("ADDV $1, R7, R7").Raw("JMP sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
