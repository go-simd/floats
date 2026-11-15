//go:build ignore

// Command gen produces cdot_amd64.s with go-asmgen: the AVX2+FMA kernel of CDotC (cdot.go).
// Per block of four complex numbers (8 float64 of each input, two Y registers): Y4, Y5 = a;
// Y6, Y7 = b; Y8, Y9 = b with the real and imaginary parts swapped inside each 128-bit lane
// (VPERMILPD $5); Y0, Y1 += a*b and Y2, Y3 += a*swap(b), one VFMADD231PD each. After the
// loop Y0 += Y1 and Y2 += Y3 are stored to out[0:4] and out[4:8]. SI = a, DX = b, R8 = out,
// R9 = blocks. Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> cdot_amd64_gen.go
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
	if err := os.WriteFile("cdot_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cdot_amd64.s")
}

func kernel() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("a"), abi.Slice("b"), abi.Slice("out")}, nil)
	b := amd64.NewFunc("cdotcAVX2", sig, 0)
	b.LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX").LoadArg("out_base", "R8").
		Raw("MOVQ CX, R9").Raw("SHRQ $3, R9")
	for k := 0; k < 4; k++ {
		b.Raw("VXORPD Y%d, Y%d, Y%d", k, k, k)
	}
	b.Raw("TESTQ R9, R9").Raw("JZ done").Label("loop").
		Raw("VMOVUPD (SI), Y4").Raw("VMOVUPD 32(SI), Y5").
		Raw("VMOVUPD (DX), Y6").Raw("VMOVUPD 32(DX), Y7").
		Raw("VPERMILPD $5, Y6, Y8").Raw("VPERMILPD $5, Y7, Y9").
		Raw("VFMADD231PD Y6, Y4, Y0").Raw("VFMADD231PD Y7, Y5, Y1").
		Raw("VFMADD231PD Y8, Y4, Y2").Raw("VFMADD231PD Y9, Y5, Y3").
		Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").Raw("DECQ R9").Raw("JNZ loop").
		Label("done").
		Raw("VADDPD Y1, Y0, Y0").Raw("VADDPD Y3, Y2, Y2").
		Raw("VMOVUPD Y0, (R8)").Raw("VMOVUPD Y2, 32(R8)").
		Raw("VZEROUPPER").Ret()
	return b
}
