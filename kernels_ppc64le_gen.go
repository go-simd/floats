//go:build ignore

// Command gen produces kernels_ppc64le.s with go-asmgen: VSX FMA reduction
// kernels for the float32/float64 dot product, sum and sum-of-squared-diffs.
//
// The released Go assembler exposes the VSX *loads* (LXVD2X for 2×f64, LXVW4X
// for 4×f32) and the VSR↔GPR moves (MFVSRD/MFVSRLD/MTVSRD), but NOT the XV*
// vector-FP arithmetic (XVMADDADP, XVADDDP, XVSUBDP and their SP forms), so
// those eight ops are emitted as their fixed Power ISA encodings via WORD. The
// encodings were taken from GNU as (-mpower9) and are commented inline; the
// register layout is pinned (acc=VS2, a=VS0, b=VS1, diff=VS3) so the constants
// are stable.
//
//	WORD $0xf0400b08  xvmaddadp vs2,vs0,vs1   vs2 += vs0*vs1   (f64)
//	WORD $0xf0401300  xvadddp   vs2,vs0,vs2   vs2 += vs0       (f64)
//	WORD $0xf0600b40  xvsubdp   vs3,vs0,vs1   vs3 = vs0-vs1    (f64)
//	WORD $0xf0431b08  xvmaddadp vs2,vs3,vs3   vs2 += vs3*vs3   (f64)
//	WORD $0xf0400a08  xvmaddasp vs2,vs0,vs1   vs2 += vs0*vs1   (f32)
//	WORD $0xf0401200  xvaddsp   vs2,vs0,vs2   vs2 += vs0       (f32)
//	WORD $0xf0600a40  xvsubsp   vs3,vs0,vs1   vs3 = vs0-vs1    (f32)
//	WORD $0xf0431a08  xvmaddasp vs2,vs3,vs3   vs2 += vs3*vs3   (f32)
//
// Each kernel folds the bulk into VS2 (a 2×f64 / 4×f32 accumulator) at the
// vector stride, then horizontally reduces and finishes the < stride tail in
// scalar FPRs. f64 extracts the two doublewords with MFVSRD/MFVSRLD, moves them
// to FPRs (MTVSRD, since VS0..31 alias F0..31) and adds; f32 spills VS2 to a
// 16-byte stack scratch with STXVW4X and adds the four lanes.
//
// Lane order: LXVD2X on little-endian loads element 0 into the high doubleword,
// but every kernel is an element-wise reduction with a commutative final
// horizontal sum, so lane numbering does not change the result. VSX is the
// POWER8 baseline, so no runtime feature gate is needed. The qemu-ppc64le
// differential tests and fuzzers are the proof.
//
// Scalar tail operand orders (verified on qemu): the Plan 9 ppc64 FMA is
// `FMADD/FMADDS A, B, C, T` → T = A*C + B (the SECOND operand is the addend,
// the third the second multiplier); `FSUB X, Y` → Y = Y − X.
//
// Run: go run kernels_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func dotSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)},
	)
}
func sumSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)},
	)
}
func dotSig32() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", abi.Float32)},
	)
}
func sumSig32() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", abi.Float32)},
	)
}

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	if err := os.WriteFile("kernels_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_ppc64le.s")
}

// foldF64 reduces VS2 (2×f64) to F0 = doubleword0 + doubleword1.
func foldF64(b *ppc64.Builder) {
	b.Raw("MFVSRD VS2, R9"). // high doubleword bits
					Raw("MFVSRLD VS2, R10"). // low doubleword bits
					Raw("MTVSRD R9, VS0").   // -> F0
					Raw("MTVSRD R10, VS1").  // -> F1
					Raw("FADD F1, F0")       // F0 += F1
}

// foldF32 spills VS2 (4×f32) to the 16-byte stack scratch at 0(R1)-relative
// offset 8 and sums the four lanes into F0. (R1 is SP; offset 8 stays clear of
// the back-chain word and within the frame.)
func foldF32(b *ppc64.Builder) {
	b.Raw("MOVD $32, R11").Raw("ADD R1, R11, R11").
		Raw("STXVW4X VS2, (R11)(R0)").
		Raw("FMOVS 0(R11), F0").
		Raw("FMOVS 4(R11), F1").
		Raw("FMOVS 8(R11), F2").
		Raw("FMOVS 12(R11), F3").
		Raw("FADDS F1, F0").
		Raw("FADDS F2, F0").
		Raw("FADDS F3, F0")
}

// ---- float64 (VS2 = 2 lanes, stride 2) ----

func dotF64() *ppc64.Builder {
	b := ppc64.NewFunc("dotVSX", dotSig(), 0)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").LoadArg("b_base", "R5").
		Raw("XXLXOR VS2, VS2, VS2"). // acc = 0
		Raw("MOVD $0, R6").          // i = 0
		Label("vloop").
		Raw("ADD $2, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $3, R6, R8").
		Raw("LXVD2X (R3)(R8), VS0").
		Raw("LXVD2X (R5)(R8), VS1").
		Raw("WORD $0xf0400b08"). // xvmaddadp vs2,vs0,vs1 : acc += a*b
		Raw("ADD $2, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $3, R6, R8").
		Raw("FMOVD (R3)(R8), F1").
		Raw("FMOVD (R5)(R8), F2").
		Raw("FMADD F1, F0, F2, F0"). // F0 += a*b
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF64() *ppc64.Builder {
	b := ppc64.NewFunc("sumVSX", sumSig(), 0)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").
		Raw("XXLXOR VS2, VS2, VS2").
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $2, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $3, R6, R8").
		Raw("LXVD2X (R3)(R8), VS0").
		Raw("WORD $0xf0401300"). // xvadddp vs2,vs0,vs2 : acc += a
		Raw("ADD $2, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $3, R6, R8").
		Raw("FMOVD (R3)(R8), F1").
		Raw("FADD F1, F0").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF64() *ppc64.Builder {
	b := ppc64.NewFunc("sumSqDiffVSX", dotSig(), 0)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").LoadArg("b_base", "R5").
		Raw("XXLXOR VS2, VS2, VS2").
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $2, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $3, R6, R8").
		Raw("LXVD2X (R3)(R8), VS0").
		Raw("LXVD2X (R5)(R8), VS1").
		Raw("WORD $0xf0600b40"). // xvsubdp vs3,vs0,vs1 : d = a-b
		Raw("WORD $0xf0431b08"). // xvmaddadp vs2,vs3,vs3 : acc += d*d
		Raw("ADD $2, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF64(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $3, R6, R8").
		Raw("FMOVD (R3)(R8), F1").
		Raw("FMOVD (R5)(R8), F2").
		Raw("FSUB F2, F1"). // F1 = a-b
		Raw("FMADD F1, F0, F1, F0").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// ---- float32 (VS2 = 4 lanes, stride 4); $32 frame for the spill scratch ----

func dotF32() *ppc64.Builder {
	b := ppc64.NewFunc("dot32VSX", dotSig32(), 48)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").LoadArg("b_base", "R5").
		Raw("XXLXOR VS2, VS2, VS2").
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $4, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $2, R6, R8").
		Raw("LXVW4X (R3)(R8), VS0").
		Raw("LXVW4X (R5)(R8), VS1").
		Raw("WORD $0xf0400a08"). // xvmaddasp vs2,vs0,vs1
		Raw("ADD $4, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $2, R6, R8").
		Raw("FMOVS (R3)(R8), F1").
		Raw("FMOVS (R5)(R8), F2").
		Raw("FMADDS F1, F0, F2, F0").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF32() *ppc64.Builder {
	b := ppc64.NewFunc("sum32VSX", sumSig32(), 48)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").
		Raw("XXLXOR VS2, VS2, VS2").
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $4, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $2, R6, R8").
		Raw("LXVW4X (R3)(R8), VS0").
		Raw("WORD $0xf0401200"). // xvaddsp vs2,vs0,vs2
		Raw("ADD $4, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $2, R6, R8").
		Raw("FMOVS (R3)(R8), F1").
		Raw("FADDS F1, F0").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF32() *ppc64.Builder {
	b := ppc64.NewFunc("sumSqDiff32VSX", dotSig32(), 48)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4").LoadArg("b_base", "R5").
		Raw("XXLXOR VS2, VS2, VS2").
		Raw("MOVD $0, R6").
		Label("vloop").
		Raw("ADD $4, R6, R7").Raw("CMP R7, R4").Raw("BGT vtail").
		Raw("SLD $2, R6, R8").
		Raw("LXVW4X (R3)(R8), VS0").
		Raw("LXVW4X (R5)(R8), VS1").
		Raw("WORD $0xf0600a40"). // xvsubsp vs3,vs0,vs1
		Raw("WORD $0xf0431a08"). // xvmaddasp vs2,vs3,vs3
		Raw("ADD $4, R6, R6").Raw("BR vloop").
		Label("vtail")
	foldF32(b)
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("SLD $2, R6, R8").
		Raw("FMOVS (R3)(R8), F1").
		Raw("FMOVS (R5)(R8), F2").
		Raw("FSUBS F2, F1").
		Raw("FMADDS F1, F0, F1, F0").
		Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
