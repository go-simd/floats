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
	// block emits one four-complex step with accumulators acc[0..3], reading at
	// byte offset off.
	block := func(off int, acc [4]string) {
		b.Raw("VMOVUPD %d(SI), Y4", off).Raw("VMOVUPD %d(SI), Y5", off+32).
			Raw("VMOVUPD %d(DX), Y6", off).Raw("VMOVUPD %d(DX), Y7", off+32).
			Raw("VPERMILPD $5, Y6, Y8").Raw("VPERMILPD $5, Y7, Y9").
			Raw("VFMADD231PD Y6, Y4, %s", acc[0]).Raw("VFMADD231PD Y7, Y5, %s", acc[1]).
			Raw("VFMADD231PD Y8, Y4, %s", acc[2]).Raw("VFMADD231PD Y9, Y5, %s", acc[3])
	}
	// Each accumulator is a loop-carried FMA chain (latency 4, two FMA ports),
	// so one block per pass keeps only four chains in flight: two blocks per
	// pass run eight (Y0..Y3 and Y10..Y13), summed after the loop. The summation
	// order is therefore not the left-to-right one (bound: gamma_n sum|a_i b_i|).
	b.Raw("TESTQ R9, R9").Raw("JZ done").
		Raw("CMPQ R9, $2").Raw("JLT loop")
	for k := 10; k < 14; k++ {
		b.Raw("VXORPD Y%d, Y%d, Y%d", k, k, k)
	}
	b.Label("loop2")
	block(0, [4]string{"Y0", "Y1", "Y2", "Y3"})
	block(64, [4]string{"Y10", "Y11", "Y12", "Y13"})
	b.Raw("ADDQ $128, SI").Raw("ADDQ $128, DX").Raw("SUBQ $2, R9").
		Raw("CMPQ R9, $2").Raw("JGE loop2").
		Raw("VADDPD Y10, Y0, Y0").Raw("VADDPD Y11, Y1, Y1").
		Raw("VADDPD Y12, Y2, Y2").Raw("VADDPD Y13, Y3, Y3").
		Raw("TESTQ R9, R9").Raw("JZ done").
		Label("loop")
	block(0, [4]string{"Y0", "Y1", "Y2", "Y3"})
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").Raw("DECQ R9").Raw("JNZ loop").
		Label("done").
		Raw("VADDPD Y1, Y0, Y0").Raw("VADDPD Y3, Y2, Y2").
		Raw("VMOVUPD Y0, (R8)").Raw("VMOVUPD Y2, 32(R8)").
		Raw("VZEROUPPER").Ret()
	return b
}
