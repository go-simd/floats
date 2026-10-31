//go:build ignore

// Command gen produces kernels_arm64.s with go-asmgen: NEON FMA reduction
// kernels for the float32/float64 dot product, sum and sum-of-squared-diffs,
// plus the float64 elementwise updates Axpy, Xpby, AxpyDot and Scale (see the
// section at the end of this file).
//
// The bulk loops fold with VFMLA (acc += a*b lane-wise). The sum kernels
// accumulate against a broadcast 1.0 vector (acc += a*1, which rounds exactly
// like acc + a); they were written against an assembler with no vector-FP add.
// The Go 1.27 assembler has VFADD/VFSUB/VFADDP. The squared-difference kernels
// form d = a−b exactly (VFSUB for float32; VFMLS against 1.0 for float64, see
// ssdF64) before VFMLA d*d — which avoids the catastrophic cancellation of the
// Σa²−2Σab+Σb² expansion on the distance hot path.
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

// axpySig is func(<scalar> float64, x, y []float64).
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
	f := emit.NewFile("arm64")
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

// ssdF64 accumulates Σ(a−b)². The difference is formed with VFMLS against a
// broadcast 1.0: copy a into V7, then V7 -= b*1 gives V7 = a−b exactly (×1 is
// exact, one rounding), and VFMLA V7,V7,acc adds d². This avoids the
// catastrophic cancellation that the Σa²−2Σab+Σb² expansion would suffer when
// a≈b — the distance hot path.
//
// VFSUB (available since the Go 1.27 assembler) gives the same d bit for bit
// and is what ssdF32 uses, but swapping it in here measured Distance/8 about 9%
// slower on an Apple M4 Max. Padding the swapped kernel back to the old layout
// with NOPs removed the gap, so the cost is code placement, not the
// instruction; it stays VFMLS until that is addressed.
//
// Aligning the swapped loop does not address it. VFSUB with PCALIGN $16 or $32
// before vloop (2026-10-07, M4 Max, Go 1.27.1, 12 runs alternating with this
// kernel, load below 8, medians) measured Distance/8 +11.8% and +10.8% (12.01 →
// 13.43 and 13.30 ns; +12.5% and +11.7% in a second series of 10), against
// +6.5% for VFSUB unaligned. /64 was +1.6% and +0.6%, /512 and /4096 within
// ±0.7%. Results were bit identical over 1e6 random inputs (NaN payloads
// aside), so only speed kept it out. This loop starts at 12 mod 16; a 16- or
// 32-byte aligned one is slower here.
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
		Raw("MOVD $0, R3").
		Label("vloop").
		Raw("ADD $4, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<2, R0, R5").Raw("VLD1 (R5), [V1.S4]").
		Raw("ADD R3<<2, R2, R6").Raw("VLD1 (R6), [V2.S4]").
		Raw("VFSUB V2.S4, V1.S4, V7.S4"). // d = a-b
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

// ---- float64 elementwise update kernels (BLAS level 1) ----
//
// Register plan: V31.D2 = the broadcast scalar (F31 keeps it for the scalar
// tail), R0 = x, R1 = n, R2 = y, R7 = w, R3 = i, R5/R6/R8 = element addresses.
//
// Every element is ONE fused multiply-add (VFMLA lane-wise, FMADDD in the
// tail), so each result is bit-identical to math.FMA(alpha, x[i], y[i]) (or
// math.FMA(beta, y[i], x[i]) for Xpby). The bulk loop is unrolled ×4: one
// four-register VLD1/VST1 moves 8 elements per array per iteration. A 2-wide
// loop and a scalar tail finish the remainder. Every access is at an index < n:
// the unrolled loop runs only while i+8 <= n, the 2-wide one while i+2 <= n, the
// tail is scalar — so no kernel touches memory past the end of a slice.
//
// Operand orders: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm; `FMADDD Fm, Fa, Fn, Fd` is
// Fd = Fn*Fm + Fa; `VFMUL Vm, Vn, Vd` is Vd = Vn*Vm.

const (
	xs = "[V1.D2, V2.D2, V3.D2, V4.D2]"
	ys = "[V5.D2, V6.D2, V7.D2, V8.D2]"
	ws = "[V9.D2, V10.D2, V11.D2, V12.D2]"
)

func loadScalar(b *arm64.Builder, name string) {
	b.LoadArg(name, "F31").Raw("VDUP V31.D[0], V31.D2")
}

// axpyF64: y[i] = fma(alpha, x[i], y[i]).
func axpyF64() *arm64.Builder {
	b := arm64.NewFunc("axpyNEON", axpySig("alpha"), 0)
	loadScalar(b, "alpha")
	b.LoadArg("x_base", "R0").LoadArg("x_len", "R1").LoadArg("y_base", "R2").
		Raw("MOVD $0, R3").
		Label("uloop").
		Raw("ADD $8, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<3, R0, R5").Raw("ADD R3<<3, R2, R6").
		Raw("VLD1 (R5), " + xs).
		Raw("VLD1 (R6), " + ys)
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMLA V31.D2, V%d.D2, V%d.D2", 1+k, 5+k)) // y += x*alpha
	}
	b.Raw("VST1 " + ys + ", (R6)").
		Raw("ADD $8, R3, R3").Raw("B uloop").
		Label("vloop").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT sloop").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("ADD R3<<3, R2, R6").Raw("VLD1 (R6), [V5.D2]").
		Raw("VFMLA V31.D2, V1.D2, V5.D2").
		Raw("VST1 [V5.D2], (R6)").
		Raw("ADD $2, R3, R3").Raw("B vloop").
		Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMOVD (R2)(R3<<3), F2").
		Raw("FMADDD F31, F2, F1, F2"). // F2 = x*alpha + y
		Raw("FMOVD F2, (R2)(R3<<3)").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").Ret()
	return b
}

// xpbyF64: y[i] = fma(beta, y[i], x[i]). x is the addend, so the result forms
// in the x registers and is stored to y.
func xpbyF64() *arm64.Builder {
	b := arm64.NewFunc("xpbyNEON", axpySig("beta"), 0)
	loadScalar(b, "beta")
	b.LoadArg("x_base", "R0").LoadArg("x_len", "R1").LoadArg("y_base", "R2").
		Raw("MOVD $0, R3").
		Label("uloop").
		Raw("ADD $8, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<3, R0, R5").Raw("ADD R3<<3, R2, R6").
		Raw("VLD1 (R5), " + xs).
		Raw("VLD1 (R6), " + ys)
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMLA V31.D2, V%d.D2, V%d.D2", 5+k, 1+k)) // x += y*beta
	}
	b.Raw("VST1 " + xs + ", (R6)").
		Raw("ADD $8, R3, R3").Raw("B uloop").
		Label("vloop").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT sloop").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("ADD R3<<3, R2, R6").Raw("VLD1 (R6), [V5.D2]").
		Raw("VFMLA V31.D2, V5.D2, V1.D2").
		Raw("VST1 [V1.D2], (R6)").
		Raw("ADD $2, R3, R3").Raw("B vloop").
		Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMOVD (R2)(R3<<3), F2").
		Raw("FMADDD F31, F1, F2, F1"). // F1 = y*beta + x
		Raw("FMOVD F1, (R2)(R3<<3)").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").Ret()
	return b
}

// axpyDotF64: y[i] = fma(alpha, x[i], y[i]); returns Σ y[i]*w[i] over the
// updated y in the same pass. Four D2 accumulators (V16..V19, one per unrolled
// register) are summed pairwise with VFADD, the 2-wide loop extends V16, its two
// lanes are folded through GPRs (as foldD2 does) and the scalar tail finishes.
func axpyDotF64() *arm64.Builder {
	b := arm64.NewFunc("axpyDotNEON", axpyDotSig(), 0)
	loadScalar(b, "alpha")
	b.LoadArg("x_base", "R0").LoadArg("x_len", "R1").LoadArg("y_base", "R2").LoadArg("w_base", "R7").
		Raw("VEOR V16.B16, V16.B16, V16.B16").
		Raw("VEOR V17.B16, V17.B16, V17.B16").
		Raw("VEOR V18.B16, V18.B16, V18.B16").
		Raw("VEOR V19.B16, V19.B16, V19.B16").
		Raw("MOVD $0, R3").
		Label("uloop").
		Raw("ADD $8, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<3, R0, R5").Raw("ADD R3<<3, R2, R6").Raw("ADD R3<<3, R7, R8").
		Raw("VLD1 (R5), " + xs).
		Raw("VLD1 (R6), " + ys)
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMLA V31.D2, V%d.D2, V%d.D2", 1+k, 5+k)) // y += x*alpha
	}
	// w is loaded only after y is stored, so w may alias y (r·r in CG).
	b.Raw("VST1 " + ys + ", (R6)").
		Raw("VLD1 (R8), " + ws)
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", 9+k, 5+k, 16+k)) // acc += y*w
	}
	b.Raw("ADD $8, R3, R3").Raw("B uloop").
		Label("vloop").
		Raw("VFADD V17.D2, V16.D2, V16.D2").
		Raw("VFADD V19.D2, V18.D2, V18.D2").
		Raw("VFADD V18.D2, V16.D2, V16.D2").
		Label("vloop2").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT vtail").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("ADD R3<<3, R2, R6").Raw("VLD1 (R6), [V5.D2]").
		Raw("VFMLA V31.D2, V1.D2, V5.D2").
		Raw("VST1 [V5.D2], (R6)").
		Raw("ADD R3<<3, R7, R8").Raw("VLD1 (R8), [V9.D2]").
		Raw("VFMLA V9.D2, V5.D2, V16.D2").
		Raw("ADD $2, R3, R3").Raw("B vloop2").
		Label("vtail").
		Raw("VMOV V16.D[0], R9").Raw("FMOVD R9, F0").
		Raw("VMOV V16.D[1], R9").Raw("FMOVD R9, F1").
		Raw("FADDD F1, F0, F0").
		Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMOVD (R2)(R3<<3), F2").
		Raw("FMADDD F31, F2, F1, F2"). // y = x*alpha + y
		Raw("FMOVD F2, (R2)(R3<<3)").
		Raw("FMOVD (R7)(R3<<3), F3"). // after the store: w may alias y
		Raw("FMADDD F3, F0, F2, F0"). // acc += y*w
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// scaleF64: x[i] *= alpha (one rounding, like the scalar multiply).
func scaleF64() *arm64.Builder {
	b := arm64.NewFunc("scaleNEON", scaleSig(), 0)
	loadScalar(b, "alpha")
	b.LoadArg("x_base", "R0").LoadArg("x_len", "R1").
		Raw("MOVD $0, R3").
		Label("uloop").
		Raw("ADD $8, R3, R4").Raw("CMP R1, R4").Raw("BGT vloop").
		Raw("ADD R3<<3, R0, R5").
		Raw("VLD1 (R5), " + xs)
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMUL V31.D2, V%d.D2, V%d.D2", 1+k, 1+k))
	}
	b.Raw("VST1 " + xs + ", (R5)").
		Raw("ADD $8, R3, R3").Raw("B uloop").
		Label("vloop").
		Raw("ADD $2, R3, R4").Raw("CMP R1, R4").Raw("BGT sloop").
		Raw("ADD R3<<3, R0, R5").Raw("VLD1 (R5), [V1.D2]").
		Raw("VFMUL V31.D2, V1.D2, V1.D2").
		Raw("VST1 [V1.D2], (R5)").
		Raw("ADD $2, R3, R3").Raw("B vloop").
		Label("sloop").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F1").
		Raw("FMULD F31, F1, F1").
		Raw("FMOVD F1, (R0)(R3<<3)").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").Ret()
	return b
}

// ---- Mul, Axpby3, Clamp, MaxAbs ----
//
// One loop skeleton (neonElem) for the three-operand kernels: input j is loaded
// into V(1+4j)..V(4+4j) (four registers per iteration, 8 elements per array), the
// result forms in input 0's registers and is stored to dst; a 2-wide loop (one
// register) and a 1-element loop finish the remainder. The 1-element loop loads
// and stores D1 (one double; the load zeroes the upper half of the register), so
// the same vector instructions serve every stage and no scalar FP code exists to
// diverge from them. Every access is at an index < n; every block is loaded,
// computed and stored before the next, so dst may be the same slice as any input.
//
// Registers: R0/R7/R12 = input 0/1/2, R2 = dst, R1 = n, R3 = i, R4 = bound
// check, R5/R8/R10 = input addresses, R6 = dst address; V29..V31 = broadcast
// scalars, V16.. = masks.

var neonStages = []struct {
	label, next string
	elems, regs int
	arr         string
}{
	{"uloop", "vloop", 8, 4, "D2"},
	{"vloop", "sloop", 2, 1, "D2"},
	{"sloop", "done", 1, 1, "D1"},
}

func vregs(first, n int, arr string) string {
	s := "["
	for k := 0; k < n; k++ {
		if k > 0 {
			s += ", "
		}
		s += fmt.Sprintf("V%d.%s", first+k, arr)
	}
	return s + "]"
}

// neonElem emits the loop: len(ptrs) inputs, compute(k) turns register k of
// input 0 into the result.
func neonElem(b *arm64.Builder, ptrs []string, compute func(k int)) {
	addr := []string{"R5", "R8", "R10"}
	b.Raw("MOVD $0, R3")
	for _, s := range neonStages {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R3, R4", s.elems)).Raw("CMP R1, R4").Raw("BGT " + s.next)
		for j, p := range ptrs {
			b.Raw(fmt.Sprintf("ADD R3<<3, %s, %s", p, addr[j])).
				Raw(fmt.Sprintf("VLD1 (%s), %s", addr[j], vregs(1+4*j, s.regs, s.arr)))
		}
		for k := 0; k < s.regs; k++ {
			compute(k)
		}
		b.Raw("ADD R3<<3, R2, R6").
			Raw(fmt.Sprintf("VST1 %s, (R6)", vregs(1, s.regs, s.arr))).
			Raw(fmt.Sprintf("ADD $%d, R3, R3", s.elems)).Raw("B " + s.label)
	}
	b.Label("done").Ret()
}

// mulSig is func(dst, a, b []float64).
func mulSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
}

// mulF64: dst[i] = a[i]*b[i] (VFMUL: one IEEE rounding).
func mulF64() *arm64.Builder {
	b := arm64.NewFunc("mulNEON", mulSig(), 0)
	b.LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R7")
	neonElem(b, []string{"R0", "R7"}, func(k int) {
		b.Raw(fmt.Sprintf("VFMUL V%d.D2, V%d.D2, V%d.D2", 5+k, 1+k, 1+k))
	})
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

// axpby3F64: dst[i] = fma(gamma, c[i], fma(beta, b[i], alpha*a[i])): a plain
// VFMUL, then two VFMLA (Vd += Vn*Vm), bit-identical to the math.FMA chain.
func axpby3F64() *arm64.Builder {
	b := arm64.NewFunc("axpby3NEON", axpby3Sig(), 0)
	b.LoadArg("alpha", "F29").Raw("VDUP V29.D[0], V29.D2").
		LoadArg("beta", "F30").Raw("VDUP V30.D[0], V30.D2").
		LoadArg("gamma", "F31").Raw("VDUP V31.D[0], V31.D2").
		LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		LoadArg("b_base", "R7").LoadArg("c_base", "R12")
	neonElem(b, []string{"R0", "R7", "R12"}, func(k int) {
		b.Raw(fmt.Sprintf("VFMUL V29.D2, V%d.D2, V%d.D2", 1+k, 1+k)).
			Raw(fmt.Sprintf("VFMLA V30.D2, V%d.D2, V%d.D2", 5+k, 1+k)).
			Raw(fmt.Sprintf("VFMLA V31.D2, V%d.D2, V%d.D2", 9+k, 1+k))
	})
	return b
}

// clampSig is func(dst, v, lo, hi []float64).
func clampSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("v"), abi.Slice("lo"), abi.Slice("hi")}, nil)
}

// clampF64: r = v; r = lo where lo > v; r = hi where r > hi. FMAX/FMIN would
// propagate a NaN bound and order the signed zeros, which is not the contract,
// so the strict compares are done with FCMGT (false when unordered, false on
// equal zeros) and the select with BIT. `VFCMGT Vm, Vn, Vd` is Vd = Vn > Vm
// (all ones where true); `VBIT Vm, Vn, Vd` copies Vn's bits into Vd where Vm
// is set.
func clampF64() *arm64.Builder {
	b := arm64.NewFunc("clampNEON", clampSig(), 0)
	b.LoadArg("dst_base", "R2").LoadArg("v_base", "R0").LoadArg("v_len", "R1").
		LoadArg("lo_base", "R7").LoadArg("hi_base", "R12")
	neonElem(b, []string{"R0", "R7", "R12"}, func(k int) {
		v, lo, hi, m := 1+k, 5+k, 9+k, 16+k
		b.Raw(fmt.Sprintf("VFCMGT V%d.D2, V%d.D2, V%d.D2", v, lo, m)). // m = lo > v
										Raw(fmt.Sprintf("VBIT V%d.B16, V%d.B16, V%d.B16", m, lo, v)). // v = lo where m
										Raw(fmt.Sprintf("VFCMGT V%d.D2, V%d.D2, V%d.D2", hi, v, m)).  // m = v > hi
										Raw(fmt.Sprintf("VBIT V%d.B16, V%d.B16, V%d.B16", m, hi, v))  // v = hi where m
	})
	return b
}

// maxAbsSig is func(v []float64) float64.
func maxAbsSig() abi.Signature { return sumSig(abi.Float64) }

// maxAbsF64 returns max |v[i]|; VFMAX/FMAXD propagate a NaN from either operand,
// so a NaN anywhere is a NaN in the result without a side channel. Four
// accumulators (V0..V3) in the ×8 loop, V0 only afterwards; the 1-element stage
// loads D1 (upper half zero, |0| = 0 is neutral). The two lanes are folded
// through GPRs as foldD2 does.
func maxAbsF64() *arm64.Builder {
	b := arm64.NewFunc("maxAbsNEON", maxAbsSig(), 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		Raw("VEOR V0.B16, V0.B16, V0.B16").Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("VEOR V2.B16, V2.B16, V2.B16").Raw("VEOR V3.B16, V3.B16, V3.B16").
		Raw("MOVD $0, R3")
	for _, s := range neonStages {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R3, R4", s.elems)).Raw("CMP R1, R4")
		if s.label == "uloop" {
			b.Raw("BGT vcomb")
		} else {
			b.Raw("BGT " + s.next)
		}
		b.Raw("ADD R3<<3, R0, R5").
			Raw(fmt.Sprintf("VLD1 (R5), %s", vregs(4, s.regs, s.arr)))
		for k := 0; k < s.regs; k++ {
			b.Raw(fmt.Sprintf("VFABS V%d.D2, V%d.D2", 4+k, 4+k))
			if s.label == "uloop" {
				b.Raw(fmt.Sprintf("VFMAX V%d.D2, V%d.D2, V%d.D2", 4+k, k, k))
			} else {
				b.Raw(fmt.Sprintf("VFMAX V%d.D2, V0.D2, V0.D2", 4+k))
			}
		}
		b.Raw(fmt.Sprintf("ADD $%d, R3, R3", s.elems)).Raw("B " + s.label)
		if s.label == "uloop" {
			b.Label("vcomb").
				Raw("VFMAX V1.D2, V0.D2, V0.D2").Raw("VFMAX V3.D2, V2.D2, V2.D2").
				Raw("VFMAX V2.D2, V0.D2, V0.D2")
		}
	}
	b.Label("done").
		Raw("VMOV V0.D[0], R9").Raw("VMOV V0.D[1], R10"). // both lanes first: FMOVD to F0 clobbers V0
		Raw("FMOVD R9, F1").Raw("FMOVD R10, F2").
		Raw("FMAXD F2, F1, F0").
		StoreRet("F0", "ret").Ret()
	return b
}
