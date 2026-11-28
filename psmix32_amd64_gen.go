//go:build ignore

// Command gen produces psmix32_amd64.s with go-asmgen: the AVX2+FMA kernel of PSMixPlan32.Run
// (psmix32.go). One call mixes every run of a slot. Per run: the four gains of its band are
// broadcast (Y0 gLs, Y1 gLd*tg, Y2 gRs, Y3 gRd*tg, the transient gain tg folded in by one
// VMULPS each), the source of the decorrelated rows is picked by the run's source number (R10
// real plane, R11 imaginary plane), and the rows are processed eight at a time on both planes:
//
//	l = gLs*s (VMULPS), l = l + gLd'*d (VFMADD231PS); r likewise.
//
// The rows left after the full vectors of a run (1 to 7): when the run holds 8 rows or more,
// one more vector over its LAST 8 rows (the rows the previous vector already wrote are
// recomputed to the same values); when it holds fewer, one vector from its first row, which
// computes and writes up to 7 rows past the end of the run. That is the liberty the contract
// of psmix32.go names: the runs come in increasing row order, so the next run overwrites those
// rows, and the plan requires 7 elements of slack after the last run. No masked access: a
// VMASKMOVPS store is several times slower than a plain one on Zen 4.
//
// AX = the run cursor, CX = runs left, DX = byte offset of the vector (4 per row, in every
// plane), R8 = first row of the run, R9 = rows left, SI/DI = s planes, R10/R11 = d planes,
// R12/R13 = l planes, BX = scratch (and the pointer of the r plane being stored). The r planes
// are reloaded from their argument slots where they are stored (there are more pointers than
// registers worth keeping). The caller guarantees at least one run and the lengths; the gate is
// hasAVX2FMA. Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> psmix32_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(kernel().Func())
	if err := os.WriteFile("psmix32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote psmix32_amd64.s")
}

// vector emits the mixing of the 8 rows at byte offset DX of both planes.
func vector(b *amd64.Builder) {
	// real plane: s in Y4, d in Y5
	b.Raw("VMOVUPS (SI)(DX*1), Y4").Raw("VMOVUPS (R10)(DX*1), Y5").
		Raw("VMULPS Y4, Y0, Y6").Raw("VFMADD231PS Y5, Y1, Y6").
		Raw("VMULPS Y4, Y2, Y7").Raw("VFMADD231PS Y5, Y3, Y7").
		Raw("VMOVUPS Y6, (R12)(DX*1)")
	b.LoadArg("rre", "BX").Raw("VMOVUPS Y7, (BX)(DX*1)")
	// imaginary plane: s in Y8, d in Y9
	b.Raw("VMOVUPS (DI)(DX*1), Y8").Raw("VMOVUPS (R11)(DX*1), Y9").
		Raw("VMULPS Y8, Y0, Y10").Raw("VFMADD231PS Y9, Y1, Y10").
		Raw("VMULPS Y8, Y2, Y11").Raw("VFMADD231PS Y9, Y3, Y11").
		Raw("VMOVUPS Y10, (R13)(DX*1)")
	b.LoadArg("rim", "BX").Raw("VMOVUPS Y11, (BX)(DX*1)")
}

func kernel() *amd64.Builder {
	args := []abi.Arg{abi.Scalar("nruns", abi.Int64)}
	for _, n := range []string{"runs", "gains", "tg", "sre", "sim", "lre", "lim", "rre", "rim", "d0re", "d0im", "d1re", "d1im", "d2re", "d2im"} {
		args = append(args, abi.Scalar(n, abi.Int64))
	}
	b := amd64.NewFunc("psMix32AVX2", abi.LayoutArgs(args, nil), 0)
	b.LoadArg("nruns", "CX").LoadArg("runs", "AX").
		LoadArg("sre", "SI").LoadArg("sim", "DI").LoadArg("lre", "R12").LoadArg("lim", "R13").
		Label("run").
		Raw("MOVLQSX (AX), R8").Raw("MOVLQSX 4(AX), R9").Raw("SUBQ R8, R9").
		// the gains of the band: BX = 16 * band, R14 = band
		Raw("MOVLQSX 8(AX), BX").Raw("MOVQ BX, DX").Raw("SHLQ $4, BX").
		LoadArg("gains", "R14").
		Raw("VBROADCASTSS (R14)(BX*1), Y0").Raw("VBROADCASTSS 4(R14)(BX*1), Y1").
		Raw("VBROADCASTSS 8(R14)(BX*1), Y2").Raw("VBROADCASTSS 12(R14)(BX*1), Y3").
		LoadArg("tg", "R14").Raw("VBROADCASTSS (R14)(DX*4), Y4").
		Raw("VMULPS Y4, Y1, Y1").Raw("VMULPS Y4, Y3, Y3").
		// the source of the decorrelated rows
		Raw("MOVLQSX 12(AX), BX").Raw("TESTQ BX, BX").Raw("JNE src1").
		LoadArg("d0re", "R10").LoadArg("d0im", "R11").Raw("JMP srcdone").
		Label("src1").Raw("CMPQ BX, $1").Raw("JNE src2").
		LoadArg("d1re", "R10").LoadArg("d1im", "R11").Raw("JMP srcdone").
		Label("src2").LoadArg("d2re", "R10").LoadArg("d2im", "R11").
		Label("srcdone").
		Raw("MOVQ R8, DX").Raw("SHLQ $2, DX").
		Label("vec").Raw("CMPQ R9, $8").Raw("JLT tail")
	vector(b)
	b.Raw("ADDQ $32, DX").Raw("SUBQ $8, R9").Raw("JMP vec").
		Label("tail").Raw("TESTQ R9, R9").Raw("JEQ next").
		// rows left: no full vector yet (a run of fewer than 8 rows) goes forward from its
		// first row; otherwise the last 8 rows of the run are redone
		Raw("MOVQ R8, BX").Raw("SHLQ $2, BX").Raw("CMPQ DX, BX").Raw("JEQ last").
		Raw("MOVQ $8, BX").Raw("SUBQ R9, BX").Raw("SHLQ $2, BX").Raw("SUBQ BX, DX").
		Label("last")
	vector(b)
	b.Label("next").Raw("ADDQ $16, AX").Raw("DECQ CX").Raw("JNZ run").
		Raw("VZEROUPPER").Ret()
	return b
}
