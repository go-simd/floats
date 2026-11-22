//go:build ignore

// Command gen produces mulrows32_amd64.s with go-asmgen: the AVX2+FMA kernel of
// MulAddRows32 (mulrows32.go). For j < len(dst), dst[j] = FMA(a[k*lda+j], b[k*ldb+j],
// dst[j]) for k = 0 .. rows-1 in order, one VFMADD231PS (VFMADD231SS for the last
// columns) per term.
//
// Blocks of 32, 8 and 1 columns: Y0..Y(w-1) accumulate (loaded once, stored once),
// Y4.. take the a row. DI = dst, CX = len(dst), DX = a, R9 = 4*lda, SI = b, R8 = 4*ldb,
// BX = rows, AX = j. The caller guarantees rows >= 1 and len(dst) >= 1, and the gate is
// hasAVX2FMA. Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> mulrows32_amd64_gen.go
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
	if err := os.WriteFile("mulrows32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mulrows32_amd64.s")
}

func kernel() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Scalar("lda", abi.Int64),
		abi.Slice("b"), abi.Scalar("ldb", abi.Int64), abi.Scalar("rows", abi.Int64)}, nil)
	b := amd64.NewFunc("mulAddRows32AVX2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("dst_len", "CX").LoadArg("a_base", "DX").LoadArg("lda", "R9").
		LoadArg("b_base", "SI").LoadArg("ldb", "R8").LoadArg("rows", "BX").
		Raw("SHLQ $2, R9").Raw("SHLQ $2, R8").Raw("XORQ AX, AX")
	stages := []struct {
		label, next string
		cols, regs  int
	}{
		{"mloop", "mloop8", 32, 4},
		{"mloop8", "mloop1", 8, 1},
	}
	for _, s := range stages {
		b.Label(s.label).
			Raw("LEAQ %d(AX), R10", s.cols).Raw("CMPQ R10, CX").Raw("JGT " + s.next)
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPS %d(DI)(AX*4), Y%d", 32*k, k)
		}
		b.Raw("LEAQ (DX)(AX*4), R11").Raw("LEAQ (SI)(AX*4), R12").Raw("MOVQ BX, R13").
			Label(s.label + "k")
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPS %d(R11), Y%d", 32*k, 4+k).
				Raw("VFMADD231PS %d(R12), Y%d, Y%d", 32*k, 4+k, k) // acc += a*b
		}
		b.Raw("ADDQ R9, R11").Raw("ADDQ R8, R12").Raw("SUBQ $1, R13").Raw("JNE " + s.label + "k")
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOVUPS Y%d, %d(DI)(AX*4)", k, 32*k)
		}
		b.Raw("ADDQ $%d, AX", s.cols).Raw("JMP " + s.label)
	}
	b.Label("mloop1").
		Raw("CMPQ AX, CX").Raw("JGE mdone").
		Raw("VMOVSS (DI)(AX*4), X0").
		Raw("LEAQ (DX)(AX*4), R11").Raw("LEAQ (SI)(AX*4), R12").Raw("MOVQ BX, R13").
		Label("mloop1k").
		Raw("VMOVSS (R11), X4").Raw("VFMADD231SS (R12), X4, X0").
		Raw("ADDQ R9, R11").Raw("ADDQ R8, R12").Raw("SUBQ $1, R13").Raw("JNE mloop1k").
		Raw("VMOVSS X0, (DI)(AX*4)").
		Raw("ADDQ $1, AX").Raw("JMP mloop1").
		Label("mdone").Raw("VZEROUPPER").Ret()
	return b
}
