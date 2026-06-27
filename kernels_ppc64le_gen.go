//go:build ignore

// Command gen produces kernels_ppc64le.s with go-asmgen: VSX FMA reduction
// kernels for the float32 dot product, sum and sum-of-squared-diffs.
//
// Float64 has NO VSX kernel here, deliberately. On real POWER9 silicon
// (cfarm433, go1.26.4, 2026-06-27) the f64 VSX reductions are SLOWER than the
// plain Go loop the compiler already autovectorizes — measured Dot throughput
// VSX vs naive: 0.91× (n=8), 0.92× (n=64), 0.82× (n=512), 0.82× (n=4096); Sum
// and SumSqDiff land the same ~0.82×. Per the dispatch principle (never run a
// SIMD kernel where it loses to the scalar/autovectorized path) the f64 path on
// ppc64le routes to the naive autovectorizable loop instead (see
// kernels_ppc64le.go), so no f64 kernel is generated. Float32 VSX is KEPT: on
// the same POWER9 it WINS ~1.55–1.61× over naive for n>=64.
//
// The released Go assembler exposes the VSX *loads* (LXVW4X for 4×f32) and the
// VSR↔GPR moves, but NOT the XV* vector-FP arithmetic (XVMADDASP, XVADDSP,
// XVSUBSP), so those three ops are emitted as their fixed Power ISA encodings
// via WORD. The encodings were taken from GNU as (-mpower9) and are commented
// inline; the register layout is pinned (acc=VS2, a=VS0, b=VS1, diff=VS3) so the
// constants are stable.
//
//	WORD $0xf0400a08  xvmaddasp vs2,vs0,vs1   vs2 += vs0*vs1   (f32)
//	WORD $0xf0401200  xvaddsp   vs2,vs0,vs2   vs2 += vs0       (f32)
//	WORD $0xf0600a40  xvsubsp   vs3,vs0,vs1   vs3 = vs0-vs1    (f32)
//	WORD $0xf0431a08  xvmaddasp vs2,vs3,vs3   vs2 += vs3*vs3   (f32)
//
// Each kernel folds the bulk into VS2 (a 4×f32 accumulator) at the vector
// stride, then horizontally reduces and finishes the < stride tail in scalar
// FPRs by spilling VS2 to a 16-byte stack scratch with STXVW4X and adding the
// four lanes.
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
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	if err := os.WriteFile("kernels_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_ppc64le.s")
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
