//go:build ignore

// Command gen produces mulrows_amd64.s with go-asmgen: the AVX2+FMA kernel of
// MulAddRows (mulrows.go). For j < len(dst), dst[j] = FMA(a[k*lda+j], b[k*ldb+j],
// dst[j]) for k = 0 .. rows-1 in order, one VFMADD231PD (VFMADD231SD for the last
// columns) per term: the bits of the math.FMA chain.
//
// Blocks of 16, 4 and 1 columns: Y0..Y(w-1) accumulate (loaded once, stored once),
// Y4.. take the a row. DI = dst, CX = len(dst), DX = a, R9 = 8*lda, SI = b, R8 = 8*ldb,
// BX = rows, AX = j. The caller guarantees rows >= 1 and len(dst) >= 1, and the gate is
// hasAVX2FMA. Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> mulrows_amd64_gen.go
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
	if err := os.WriteFile("mulrows_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mulrows_amd64.s")
}

func kernel() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Scalar("lda", abi.Int64),
		abi.Slice("b"), abi.Scalar("ldb", abi.Int64), abi.Scalar("rows", abi.Int64)}, nil)
	b := amd64.NewFunc("mulAddRowsAVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("dst_len", "CX").LoadArg("a_base", "DX").LoadArg("lda", "R9").
		LoadArg("b_base", "SI").LoadArg("ldb", "R8").LoadArg("rows", "BX").
		Raw("SHLQ $3, R9").Raw("SHLQ $3, R8").Raw("XORQ AX, AX")
	stages := []struct {
		label, next string
		cols, regs  int
	}{
		{"mloop", "mloop4", 16, 4},
		{"mloop4", "mloop1", 4, 1},
	}
	for _, s := range stages {
		b.Label(s.label).
			Raw("LEAQ %d(AX), R10", s.cols).Raw("CMPQ R10, CX").Raw("JGT " + s.next)
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPD %d(DI)(AX*8), Y%d", 32*k, k)
		}
		b.Raw("LEAQ (DX)(AX*8), R11").Raw("LEAQ (SI)(AX*8), R12").Raw("MOVQ BX, R13").
			Label(s.label + "k")
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPD %d(R11), Y%d", 32*k, 4+k).
				Raw("VFMADD231PD %d(R12), Y%d, Y%d", 32*k, 4+k, k) // acc += a*b
		}
		b.Raw("ADDQ R9, R11").Raw("ADDQ R8, R12").Raw("SUBQ $1, R13").Raw("JNE " + s.label + "k")
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPD Y%d, %d(DI)(AX*8)", k, 32*k)
		}
		b.Raw("ADDQ $%d, AX", s.cols).Raw("JMP " + s.label)
	}
	b.Label("mloop1").
		Raw("CMPQ AX, CX").Raw("JGE mdone").
		Raw("VMOVSD (DI)(AX*8), X0").
		Raw("LEAQ (DX)(AX*8), R11").Raw("LEAQ (SI)(AX*8), R12").Raw("MOVQ BX, R13").
		Label("mloop1k").
		Raw("VMOVSD (R11), X4").Raw("VFMADD231SD (R12), X4, X0").
		Raw("ADDQ R9, R11").Raw("ADDQ R8, R12").Raw("SUBQ $1, R13").Raw("JNE mloop1k").
		Raw("VMOVSD X0, (DI)(AX*8)").
		Raw("ADDQ $1, AX").Raw("JMP mloop1").
		Label("mdone").Raw("VZEROUPPER").Ret()
	return b
}
