//go:build ignore

// Command gen produces maxabsmul_amd64.s with go-asmgen: the AVX2 kernel of
// MaxAbsMul (maxabsmul.go), max_i |v[i]|*w[i] with the semantics of C's fmax
// folded from +0 (a NaN product loses).
//
// Per element: VANDPD with the sign-clearing mask (Y15, built in registers), then
// VMULPD (one rounding, like the scalar product). VMAXPD returns its SECOND
// source (the first Plan 9 operand) when either is a NaN, so `VMAXPD acc, p, acc`
// is p > acc ? p : acc: a NaN product p leaves acc, and acc, seeded with +0, is
// never a NaN. No mask is needed. Four accumulators (Y0..Y3) hide the VMAXPD
// latency, folded before the 4-wide loop and the scalar tail (a VEX.128
// instruction zeroes the upper half, so nothing vector-wide may be live then).
// The kernel returns the value of the maximum; its zero has no meaningful sign
// (the Go wrapper redoes an all-zero result with the scalar loop).
//
// Registers: SI = v, DX = w, CX = n, AX = i, R8 = bound check.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxabsmul_amd64_gen.go
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
	if err := os.WriteFile("maxabsmul_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxabsmul_amd64.s")
}

func mem(base string, off int) string { return fmt.Sprintf("%d(%s)", off, base) }

func kernel() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("v"), abi.Slice("w")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)})
	b := amd64.NewFunc("maxAbsMulAVX2", sig, 0)
	b.LoadArg("v_base", "SI").LoadArg("v_len", "CX").LoadArg("w_base", "DX").
		Raw("VPCMPEQD Y15, Y15, Y15").Raw("VPSRLQ $1, Y15, Y15").
		Raw("VXORPD Y0, Y0, Y0").Raw("VXORPD Y1, Y1, Y1").
		Raw("VXORPD Y2, Y2, Y2").Raw("VXORPD Y3, Y3, Y3").
		Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 16(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	for k := 0; k < 4; k++ {
		p := fmt.Sprintf("Y%d", 5+k)
		b.Raw("VANDPD %s, Y15, %s", mem("SI", 32*k), p).
			Raw("VMULPD %s, %s, %s", mem("DX", 32*k), p, p).
			Raw("VMAXPD Y%d, %s, Y%d", k, p, k)
	}
	b.Raw("ADDQ $16, AX").Raw("ADDQ $128, SI").Raw("ADDQ $128, DX").Raw("JMP uloop").
		Label("vloop").
		Raw("VMAXPD Y1, Y0, Y0").Raw("VMAXPD Y3, Y2, Y2").Raw("VMAXPD Y2, Y0, Y0").
		Label("vloop1").
		Raw("LEAQ 4(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vtail").
		Raw("VANDPD (SI), Y15, Y5").
		Raw("VMULPD (DX), Y5, Y5").
		Raw("VMAXPD Y0, Y5, Y0").
		Raw("ADDQ $4, AX").Raw("ADDQ $32, SI").Raw("ADDQ $32, DX").Raw("JMP vloop1").
		Label("vtail").
		Raw("VEXTRACTF128 $1, Y0, X5").Raw("VMAXPD X5, X0, X0").
		Raw("VUNPCKHPD X0, X0, X5").Raw("VMAXSD X5, X0, X0").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done").
		Raw("VMOVSD (SI), X5").
		Raw("VANDPD X15, X5, X5").
		Raw("VMULSD (DX), X5, X5").
		Raw("VMAXSD X0, X5, X0").
		Raw("ADDQ $1, AX").Raw("ADDQ $8, SI").Raw("ADDQ $8, DX").Raw("JMP sloop").
		Label("done").
		Raw("VZEROUPPER").
		StoreRet("X0", "ret").Ret()
	return b
}
