//go:build ignore

// Command gen produces psmix32_arm64.s with go-asmgen: the NEON kernel of PSMixPlan32.Run
// (psmix32.go), the shape of the AVX2 kernel at four rows per vector. Per run: the four gains
// of its band are broadcast (V0 gLs, V1 gLd*tg, V2 gRs, V3 gRd*tg, the transient gain folded
// in by one VFMUL each), the source of the decorrelated rows is picked by the run's source
// number (R9 real plane, R10 imaginary plane), and the rows are processed four at a time:
//
//	l = gLs*s (VFMUL), l += gLd'*d (VFMLA); r likewise.
//
// The rows left after the full vectors of a run (1 to 3): when the run holds 4 rows or more,
// one more vector over its LAST 4 rows (recomputed to the same values); when it holds fewer,
// one vector from its first row, which computes and writes up to 3 rows past the end of the
// run, which the contract of psmix32.go allows (up to 7, the AVX2 figure): the runs come in
// increasing row order, so the next run overwrites those rows. NEON has no masked store.
//
// R0 = the run cursor, R1 = runs left, R2 = byte offset of the vector (4 per row, in every
// plane), R3 = first row of the run, R4 = rows left, R5/R6 = s planes, R7/R8 = l planes,
// R9/R10 = d planes, R11 to R13 = scratch. The r planes are reloaded from their argument slots
// where they are stored. The caller guarantees at least one run and the lengths. Operand order:
// `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> psmix32_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(kernel().Func())
	if err := os.WriteFile("psmix32_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote psmix32_arm64.s")
}

// vector emits the mixing of the 4 rows at byte offset R2 of both planes.
func vector(b *arm64.Builder) {
	plane := func(sp, dp, lp string, rarg string, sv, dv, lv, rv int) {
		b.Raw("ADD R2, %s, R11", sp).Raw("VLD1 (R11), [V%d.S4]", sv).
			Raw("ADD R2, %s, R11", dp).Raw("VLD1 (R11), [V%d.S4]", dv).
			Raw("VFMUL V%d.S4, V0.S4, V%d.S4", sv, lv).Raw("VFMLA V%d.S4, V1.S4, V%d.S4", dv, lv).
			Raw("VFMUL V%d.S4, V2.S4, V%d.S4", sv, rv).Raw("VFMLA V%d.S4, V3.S4, V%d.S4", dv, rv)
		b.Raw("ADD R2, %s, R11", lp).Raw("VST1 [V%d.S4], (R11)", lv)
		b.LoadArg(rarg, "R11").Raw("ADD R2, R11, R11").Raw("VST1 [V%d.S4], (R11)", rv)
	}
	plane("R5", "R9", "R7", "rre", 16, 17, 18, 19)
	plane("R6", "R10", "R8", "rim", 20, 21, 22, 23)
}

func kernel() *arm64.Builder {
	args := []abi.Arg{abi.Scalar("nruns", abi.Int64)}
	for _, n := range []string{"runs", "gains", "tg", "sre", "sim", "lre", "lim", "rre", "rim", "d0re", "d0im", "d1re", "d1im", "d2re", "d2im"} {
		args = append(args, abi.Scalar(n, abi.Int64))
	}
	b := arm64.NewFunc("psMix32NEON", abi.LayoutArgs(args, nil), 0)
	b.LoadArg("nruns", "R1").LoadArg("runs", "R0").
		LoadArg("sre", "R5").LoadArg("sim", "R6").LoadArg("lre", "R7").LoadArg("lim", "R8").
		Label("run").
		Raw("MOVW (R0), R3").Raw("MOVW 4(R0), R4").Raw("SUB R3, R4, R4").
		// the gains of the band at gains + 16*band, then its transient gain at tg + 4*band
		Raw("MOVW 8(R0), R11").Raw("LSL $4, R11, R12").LoadArg("gains", "R13").Raw("ADD R12, R13, R12").
		Raw("VLD1R.P 4(R12), [V0.S4]").Raw("VLD1R.P 4(R12), [V1.S4]").
		Raw("VLD1R.P 4(R12), [V2.S4]").Raw("VLD1R (R12), [V3.S4]").
		Raw("LSL $2, R11, R12").LoadArg("tg", "R13").Raw("ADD R12, R13, R12").Raw("VLD1R (R12), [V4.S4]").
		Raw("VFMUL V4.S4, V1.S4, V1.S4").Raw("VFMUL V4.S4, V3.S4, V3.S4").
		// the source of the decorrelated rows
		Raw("MOVW 12(R0), R11").Raw("CBNZ R11, src1").
		LoadArg("d0re", "R9").LoadArg("d0im", "R10").Raw("B srcdone").
		Label("src1").Raw("CMP $1, R11").Raw("BNE src2").
		LoadArg("d1re", "R9").LoadArg("d1im", "R10").Raw("B srcdone").
		Label("src2").LoadArg("d2re", "R9").LoadArg("d2im", "R10").
		Label("srcdone").
		Raw("LSL $2, R3, R2").
		Label("vec").Raw("CMP $4, R4").Raw("BLT tail")
	vector(b)
	b.Raw("ADD $16, R2, R2").Raw("SUB $4, R4, R4").Raw("B vec").
		Label("tail").Raw("CBZ R4, next").
		// rows left: no full vector yet (a run of fewer than 4 rows) goes forward from its
		// first row; otherwise the last 4 rows of the run are redone
		Raw("LSL $2, R3, R11").Raw("CMP R11, R2").Raw("BEQ last").
		Raw("MOVD $4, R11").Raw("SUB R4, R11, R11").Raw("LSL $2, R11, R11").Raw("SUB R11, R2, R2").
		Label("last")
	vector(b)
	b.Label("next").Raw("ADD $16, R0, R0").Raw("SUB $1, R1, R1").Raw("CBNZ R1, run").Ret()
	return b
}
