//go:build ignore

// Command gen produces kernels_s390x.s with go-asmgen: z/Architecture
// vector-facility FMA reduction kernels for the float64 dot product, sum and
// sum-of-squared-differences.
//
// Only float64 is vectorised: the released assembler exposes the double-element
// vector-FP ops (VFMADB multiply-add, VFADB add, VFSDB subtract, VFMDB
// multiply) but not their single-element ".SB" counterparts, so the float32
// reductions stay on the lane-blocked scalar reference (see kernels_s390x.go).
//
// Each kernel folds the bulk into a V0 accumulator (2×f64 lanes) at stride 2,
// then horizontally reduces by extracting both lanes to GPRs (VLGVG), moving
// them to FPRs (LDGR) and adding (FADD); a scalar tail finishes the odd element
// with FMADD/FADD.
//
// Big-endian note: s390x is big-endian, but every operation here is an
// element-wise FP reduction (per-lane FMA followed by a commutative horizontal
// sum of the two lanes), so the lane numbering does not affect the result —
// VLGVG $0 and $1 pull both lanes and they are added together regardless of
// which holds which input. The differential tests and fuzzers under qemu-s390x
// (a big-endian target) are the proof.
//
// The vector facility is the z13 baseline, so no runtime feature gate is needed.
//
// Operand orders (verified against the released assembler + qemu):
//
//	VFMADB Va, Vb, Vc, Vd   Vd = Va*Vb + Vc
//	VFMDB  Va, Vb, Vd       Vd = Va*Vb
//	VFSDB  Va, Vb, Vd       Vd = Vb - Va
//	FADD   Fs, Fd           Fd = Fd + Fs
//	FMADD  Fa, Fb, Fd       Fd = Fd + Fa*Fb
//
// Run: go run kernels_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
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

func main() {
	f := emit.NewFile("s390x")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	if err := os.WriteFile("kernels_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_s390x.s")
}

// foldV0 reduces V0 (2×f64) to F0 = lane0 + lane1.
func foldV0(b *s390x.Builder) {
	b.Raw("VLGVG $0, V0, R7").
		Raw("VLGVG $1, V0, R8").
		Raw("LDGR R7, F0").
		Raw("LDGR R8, F1").
		Raw("FADD F1, F0") // F0 += F1
}

// Register plan: R1 a_base, R2 a_len, R3 b_base (LoadArg); R4 i; R5/R6 addr;
// V0 accumulator; V1/V2 chunks; V3 diff.

func dotF64() *s390x.Builder {
	b := s390x.NewFunc("dotVX", dotSig(), 0)
	b.LoadArg("a_base", "R1").LoadArg("a_len", "R2").LoadArg("b_base", "R3").
		Raw("VZERO V0").
		Raw("MOVD $0, R4").
		Label("vloop").
		Raw("ADD $2, R4, R5").Raw("CMPBGT R5, R2, vtail").
		Raw("SLD $3, R4, R6").
		Raw("VL (R1)(R6*1), V1").
		Raw("VL (R3)(R6*1), V2").
		Raw("VFMADB V1, V2, V0, V0"). // V0 += a*b
		Raw("ADD $2, R4, R4").Raw("BR vloop").
		Label("vtail")
	foldV0(b)
	b.Label("sloop").
		Raw("CMPBGE R4, R2, done").
		Raw("SLD $3, R4, R6").
		Raw("FMOVD (R1)(R6*1), F1").
		Raw("FMOVD (R3)(R6*1), F2").
		Raw("FMADD F1, F2, F0"). // F0 += a*b
		Raw("ADD $1, R4, R4").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF64() *s390x.Builder {
	b := s390x.NewFunc("sumVX", sumSig(), 0)
	b.LoadArg("a_base", "R1").LoadArg("a_len", "R2").
		Raw("VZERO V0").
		Raw("MOVD $0, R4").
		Label("vloop").
		Raw("ADD $2, R4, R5").Raw("CMPBGT R5, R2, vtail").
		Raw("SLD $3, R4, R6").
		Raw("VL (R1)(R6*1), V1").
		Raw("VFADB V1, V0, V0"). // V0 += a
		Raw("ADD $2, R4, R4").Raw("BR vloop").
		Label("vtail")
	foldV0(b)
	b.Label("sloop").
		Raw("CMPBGE R4, R2, done").
		Raw("SLD $3, R4, R6").
		Raw("FMOVD (R1)(R6*1), F1").
		Raw("FADD F1, F0").
		Raw("ADD $1, R4, R4").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF64() *s390x.Builder {
	b := s390x.NewFunc("sumSqDiffVX", dotSig(), 0)
	b.LoadArg("a_base", "R1").LoadArg("a_len", "R2").LoadArg("b_base", "R3").
		Raw("VZERO V0").
		Raw("MOVD $0, R4").
		Label("vloop").
		Raw("ADD $2, R4, R5").Raw("CMPBGT R5, R2, vtail").
		Raw("SLD $3, R4, R6").
		Raw("VL (R1)(R6*1), V1").
		Raw("VL (R3)(R6*1), V2").
		Raw("VFSDB V2, V1, V3").      // V3 = a - b
		Raw("VFMADB V3, V3, V0, V0"). // V0 += d*d
		Raw("ADD $2, R4, R4").Raw("BR vloop").
		Label("vtail")
	foldV0(b)
	b.Label("sloop").
		Raw("CMPBGE R4, R2, done").
		Raw("SLD $3, R4, R6").
		Raw("FMOVD (R1)(R6*1), F1").
		Raw("FMOVD (R3)(R6*1), F2").
		Raw("FSUB F2, F1"). // F1 = F1 - F2 = a-b
		Raw("FMADD F1, F1, F0").
		Raw("ADD $1, R4, R4").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
