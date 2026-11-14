//go:build ignore

// Command gen produces mulrows_arm64.s with go-asmgen: the NEON kernel of MulAddRows
// (mulrows.go). For j < len(dst), dst[j] = FMA(a[k*lda+j], b[k*ldb+j], dst[j]) for
// k = 0 .. rows-1 in order, one fused VFMLA (FMADDD for the last column) per term: the
// bits of the math.FMA chain.
//
// Blocks of 16, 4, 2 and 1 columns. In a block of w vector registers V0..V(w-1) are the
// accumulators (loaded once, stored once), V8.. the a row, V16.. the b row. R2 = dst,
// R3 = len(dst), R0 = a, R9 = 8*lda, R7 = b, R8 = 8*ldb, R1 = rows, R4 = j. The caller
// guarantees rows >= 1 and len(dst) >= 1.
//
// Operand order: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm; `FMADDD Fm, Fa, Fn, Fd` is
// Fd = Fn*Fm + Fa.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> mulrows_arm64_gen.go
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
	if err := os.WriteFile("mulrows_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mulrows_arm64.s")
}

func vregs(first, n int) string {
	s := "["
	for k := 0; k < n; k++ {
		if k > 0 {
			s += ", "
		}
		s += fmt.Sprintf("V%d.D2", first+k)
	}
	return s + "]"
}

func kernel() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Scalar("lda", abi.Int64),
		abi.Slice("b"), abi.Scalar("ldb", abi.Int64), abi.Scalar("rows", abi.Int64)}, nil)
	b := arm64.NewFunc("mulAddRowsNEON", sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("dst_len", "R3").LoadArg("a_base", "R0").LoadArg("lda", "R9").
		LoadArg("b_base", "R7").LoadArg("ldb", "R8").LoadArg("rows", "R1").
		Raw("LSL $3, R9, R9").Raw("LSL $3, R8, R8").Raw("MOVD $0, R4")
	stages := []struct {
		label, next string
		cols, regs  int
	}{
		{"mloop", "mloop4", 16, 8},
		{"mloop4", "mloop2", 4, 2},
		{"mloop2", "mloop1", 2, 1},
	}
	for _, s := range stages {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R4, R5", s.cols)).Raw("CMP R3, R5").Raw("BGT " + s.next).
			Raw("ADD R4<<3, R2, R6").Raw("MOVD R6, R13")
		for q := 0; q < s.regs; q += 4 {
			n := min(4, s.regs-q)
			b.Raw(fmt.Sprintf("VLD1.P %d(R13), %s", 16*n, vregs(q, n)))
		}
		b.Raw("ADD R4<<3, R0, R10").Raw("ADD R4<<3, R7, R11").Raw("MOVD R1, R12").
			Label(s.label + "k").
			Raw("MOVD R10, R14").Raw("MOVD R11, R15")
		for q := 0; q < s.regs; q += 4 {
			n := min(4, s.regs-q)
			b.Raw(fmt.Sprintf("VLD1.P %d(R14), %s", 16*n, vregs(8+q, n)))
			b.Raw(fmt.Sprintf("VLD1.P %d(R15), %s", 16*n, vregs(16+q, n)))
		}
		for k := 0; k < s.regs; k++ {
			b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", 16+k, 8+k, k)) // acc += a*b
		}
		b.Raw("ADD R9, R10, R10").Raw("ADD R8, R11, R11").Raw("SUBS $1, R12, R12").Raw("BNE " + s.label + "k")
		for q := 0; q < s.regs; q += 4 {
			n := min(4, s.regs-q)
			b.Raw(fmt.Sprintf("VST1.P %s, 16*%d(R6)", vregs(q, n), n))
		}
		b.Raw(fmt.Sprintf("ADD $%d, R4, R4", s.cols)).Raw("B " + s.label)
	}
	b.Label("mloop1").
		Raw("CMP R3, R4").Raw("BGE mdone").
		Raw("FMOVD (R2)(R4<<3), F0").
		Raw("ADD R4<<3, R0, R10").Raw("ADD R4<<3, R7, R11").Raw("MOVD R1, R12").
		Label("mloop1k").
		Raw("FMOVD (R10), F8").Raw("FMOVD (R11), F16").
		Raw("FMADDD F16, F0, F8, F0"). // F0 = F8*F16 + F0
		Raw("ADD R9, R10, R10").Raw("ADD R8, R11, R11").Raw("SUBS $1, R12, R12").Raw("BNE mloop1k").
		Raw("FMOVD F0, (R2)(R4<<3)").
		Raw("ADD $1, R4, R4").Raw("B mloop1").
		Label("mdone").Ret()
	return b
}
