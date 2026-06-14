//go:build ignore

// Command gen produces kernels_arm64.s with go-asmgen: NEON FMA reduction
// kernels for the float32/float64 dot product, sum and sum-of-squared-diffs.
//
// The bulk loops fold with VFMLA (acc += a*b lane-wise) — the only vector-FP
// FMA mnemonic the released assembler exposes; the wider vector add/sub/pairwise
// ops (VFADD/VFADDP) are NOT available, so the kernels are built from VFMLA plus
// two tricks: sum accumulates against a broadcast 1.0 vector (acc += a*1), and
// the squared-difference kernel forms d=a−b with VFMLS against that same 1.0
// vector (d = a − b*1, exact) before VFMLA d*d — which avoids the catastrophic
// cancellation of the Σa²−2Σab+Σb² expansion on the distance hot path.
//
// The dot kernels keep four independent accumulators and unroll ×4 for
// instruction-level parallelism (otherwise a single 2-wide accumulator loses to
// the compiler's pipelined scalar FMADD loop on long vectors). The horizontal
// fold extracts each lane to a GPR (VMOV), reinterprets it as a float
// (FMOVD/FMOVS) and sums with scalar FADDD/FADDS; the tail finishes in F0 with
// FMADDD/FMADDS (Fd = Fn*Fm + Fa).
//
// NEON is the ARMv8-A baseline, so there is no runtime feature gate.
//
// Run: go run kernels_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
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
	f := emit.NewFile("arm64")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	if err := os.WriteFile("kernels_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_arm64.s")
}

// foldD2 reduces V0.D2 to F0 (= lane0 + lane1) using a GPR round-trip, since the
// vector-FP pairwise add isn't in the released assembler.
func foldD2(b *arm64.Builder) {
	b.Raw("VMOV V0.D[1], R9").
		Raw("FMOVD R9, F1").
		// F0 already aliases V0.D[0].
		Raw("FADDD F1, F0, F0")
}

// foldS4 reduces V0.S4 to F0 (= lane0+lane1+lane2+lane3).
func foldS4(b *arm64.Builder) {
	b.Raw("VMOV V0.S[1], R9").Raw("FMOVS R9, F1").
		Raw("VMOV V0.S[2], R10").Raw("FMOVS R10, F2").
		Raw("VMOV V0.S[3], R11").Raw("FMOVS R11, F3").
		// F0 aliases V0.S[0].
		Raw("FADDS F1, F0, F0").
		Raw("FADDS F2, F0, F0").
		Raw("FADDS F3, F0, F0")
}

// ---- float64: V0.D2 accumulator, stride 2 ----

// dotF64 keeps four independent V0/V16/V17/V18 (each D2) accumulators and
// unrolls the loop ×4 (stride 8) so four VFMLAs are in flight per iteration —
// enough instruction-level parallelism to keep the FP units busy and beat the
// compiler's single-accumulator scalar FMADD loop on wide vectors. The four
// accumulators are combined (acc0 += accK*1 via VFMLA against a broadcast 1.0)
// before the 2-lane fold; a 2-wide block loop then a scalar tail finish the
// remainder.
func dotF64() *arm64.Builder {
	b := arm64.NewFunc("dotNEON", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V16.B16, V16.B16, V16.B16").
		Raw("VEOR V17.B16, V17.B16, V17.B16").
		Raw("VEOR V18.B16, V18.B16, V18.B16").
		Raw("FMOVD $(1.0), F19").Raw("VDUP V19.D[0], V19.D2"). // ones, to combine accs
		Raw("MOVD $0, R3").
		Label("uloop"). // ×4 unrolled, stride 8
		Raw("ADD $8, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<3, R0, R5").Raw("ADD R3<<3, R2, R6").
		Raw("VLD1.P 64(R5), [V1.D2, V2.D2, V3.D2, V4.D2]").
		Raw("VLD1.P 64(R6), [V5.D2, V6.D2, V7.D2, V8.D2]").
		Raw("VFMLA V5.D2, V1.D2, V0.D2").
		Raw("VFMLA V6.D2, V2.D2, V16.D2").
		Raw("VFMLA V7.D2, V3.D2, V17.D2").
		Raw("VFMLA V8.D2, V4.D2, V18.D2").
		Raw("ADD $8, R3, R3").Raw("B uloop").
		Label("vloop"). // combine the four accumulators into V0
		Raw("VFMLA V19.D2, V16.D2, V0.D2").
		Raw("VFMLA V19.D2, V17.D2, V0.D2").
		Raw("VFMLA V19.D2, V18.D2, V0.D2").
		Label("vloop2"). // 2-wide remainder
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("ADD R3<<3, R2, R6").Raw("VLD1 (R6), [V2.D2]").
		Raw("VFMLA V2.D2, V1.D2, V0.D2").
		Raw("ADD $2, R3, R3").Raw("B vloop2").
		Label("vtail")
	foldD2(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMOVD (R2)(R3<<3), F2").
		Raw("FMADDD F1, F0, F2, F0"). // F0 = F2*F1 + F0
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF64() *arm64.Builder {
	b := arm64.NewFunc("sumNEON", sumSig(abi.Float64), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("FMOVD $(1.0), F3").Raw("VDUP V3.D[0], V3.D2"). // ones
		Raw("MOVD $0, R3").
		Label("vloop").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("VFMLA V3.D2, V1.D2, V0.D2"). // acc += a*1
		Raw("ADD $2, R3, R3").Raw("B vloop").
		Label("vtail")
	foldD2(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FADDD F1, F0, F0").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// ssdF64 accumulates Σ(a−b)². The released assembler has no vector-FP subtract
// mnemonic, so the difference is formed with VFMLS against a broadcast 1.0:
// copy a into V7, then V7 -= b*1 gives V7 = a−b exactly (×1 is exact), and
// VFMLA V7,V7,acc adds d². This avoids the catastrophic cancellation that the
// Σa²−2Σab+Σb² expansion would suffer when a≈b — the distance hot path.
func ssdF64() *arm64.Builder {
	b := arm64.NewFunc("sumSqDiffNEON", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("FMOVD $(1.0), F3").Raw("VDUP V3.D[0], V3.D2"). // ones
		Raw("MOVD $0, R3").
		Label("vloop").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("ADD R3<<3, R2, R6").Raw("VLD1 (R6), [V2.D2]").
		Raw("VMOV V1.B16, V7.B16").       // d = a
		Raw("VFMLS V3.D2, V2.D2, V7.D2"). // d -= b*1  => d = a-b
		Raw("VFMLA V7.D2, V7.D2, V0.D2"). // acc += d*d
		Raw("ADD $2, R3, R3").Raw("B vloop").
		Label("vtail")
	foldD2(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMOVD (R2)(R3<<3), F2").
		Raw("FSUBD F2, F1, F1").
		Raw("FMADDD F1, F0, F1, F0"). // F0 += d*d
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// ---- float32: V0.S4 accumulator, stride 4 ----

func dotF32() *arm64.Builder {
	b := arm64.NewFunc("dot32NEON", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V16.B16, V16.B16, V16.B16").
		Raw("VEOR V17.B16, V17.B16, V17.B16").
		Raw("VEOR V18.B16, V18.B16, V18.B16").
		Raw("FMOVS $(1.0), F19").Raw("VDUP V19.S[0], V19.S4").
		Raw("MOVD $0, R3").
		Label("uloop"). // ×4 unrolled, stride 16
		Raw("ADD $16, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<2, R0, R5").Raw("ADD R3<<2, R2, R6").
		Raw("VLD1.P 64(R5), [V1.S4, V2.S4, V3.S4, V4.S4]").
		Raw("VLD1.P 64(R6), [V5.S4, V6.S4, V7.S4, V8.S4]").
		Raw("VFMLA V5.S4, V1.S4, V0.S4").
		Raw("VFMLA V6.S4, V2.S4, V16.S4").
		Raw("VFMLA V7.S4, V3.S4, V17.S4").
		Raw("VFMLA V8.S4, V4.S4, V18.S4").
		Raw("ADD $16, R3, R3").Raw("B uloop").
		Label("vloop").
		Raw("VFMLA V19.S4, V16.S4, V0.S4").
		Raw("VFMLA V19.S4, V17.S4, V0.S4").
		Raw("VFMLA V19.S4, V18.S4, V0.S4").
		Label("vloop2"). // 4-wide remainder
		Raw("ADD $4, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<2, R0, R5").Raw("VLD1 (R5), [V1.S4]").
		Raw("ADD R3<<2, R2, R6").Raw("VLD1 (R6), [V2.S4]").
		Raw("VFMLA V2.S4, V1.S4, V0.S4").
		Raw("ADD $4, R3, R3").Raw("B vloop2").
		Label("vtail")
	foldS4(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVS (R0)(R3<<2), F1").
		Raw("FMOVS (R2)(R3<<2), F2").
		Raw("FMADDS F1, F0, F2, F0").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF32() *arm64.Builder {
	b := arm64.NewFunc("sum32NEON", sumSig(abi.Float32), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("FMOVS $(1.0), F3").Raw("VDUP V3.S[0], V3.S4").
		Raw("MOVD $0, R3").
		Label("vloop").
		Raw("ADD $4, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<2, R0, R5").Raw("VLD1 (R5), [V1.S4]").
		Raw("VFMLA V3.S4, V1.S4, V0.S4").
		Raw("ADD $4, R3, R3").Raw("B vloop").
		Label("vtail")
	foldS4(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVS (R0)(R3<<2), F1").
		Raw("FADDS F1, F0, F0").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF32() *arm64.Builder {
	b := arm64.NewFunc("sumSqDiff32NEON", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("FMOVS $(1.0), F3").Raw("VDUP V3.S[0], V3.S4"). // ones
		Raw("MOVD $0, R3").
		Label("vloop").
		Raw("ADD $4, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<2, R0, R5").Raw("VLD1 (R5), [V1.S4]").
		Raw("ADD R3<<2, R2, R6").Raw("VLD1 (R6), [V2.S4]").
		Raw("VMOV V1.B16, V7.B16").       // d = a
		Raw("VFMLS V3.S4, V2.S4, V7.S4"). // d -= b*1 => a-b
		Raw("VFMLA V7.S4, V7.S4, V0.S4"). // acc += d*d
		Raw("ADD $4, R3, R3").Raw("B vloop").
		Label("vtail")
	foldS4(b)
	b.Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVS (R0)(R3<<2), F1").
		Raw("FMOVS (R2)(R3<<2), F2").
		Raw("FSUBS F2, F1, F1").
		Raw("FMADDS F1, F0, F1, F0").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
