//go:build ignore

// Command gen produces fold_amd64.s with go-asmgen: the AVX2 kernels of the MDCT fold
// and unfold (fold.go), four lanes of float64 per step; the blocks are those of
// fold_arm64_gen.go with four lanes instead of two.
//
// A register is reversed with VPERMPD $0x1B, a float32 quad with VPERMILPS $0x1B
// (the whole XMM register); the negation is an XOR with the sign bit (built with
// VPCMPEQD and VPSLLQ $63), so -(+0) is -0 as in the portable loop. The float32
// forms widen the input with VCVTPS2PD (the sum is float64) and narrow the output
// with VCVTPD2PS. No FMA is used: every output is one addition, subtraction or
// negation, the portable loop's result exactly; the gate is hasAVX2FMA like the other
// AVX2 kernels, so the CPU set is the same. The kernels do floor(h/4) steps; the Go
// caller finishes the tail.
//
// Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`: VSUBPD Y2, Y1, Y3 is
// Y3 = Y1 - Y2.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> fold_amd64_gen.go
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
	f.Add(fold("mdctFold64AVX2", false).Func())
	f.Add(fold("mdctFoldFrom32AVX2", true).Func())
	f.Add(unfold("mdctUnfold64AVX2", false).Func())
	f.Add(unfold("mdctUnfoldTo32AVX2", true).Func())
	if err := os.WriteFile("fold_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote fold_amd64.s")
}

// fold: func(dst, x []float64), x []float32 when in32.
func fold(name string, in32 bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// SI C front, R8 D back, R9 A front, R10 B back, DI dst[i], DX dst[h+i], R11 steps
	b.LoadArg("dst_base", "DI").LoadArg("x_base", "SI").LoadArg("x_len", "CX")
	size, xstep := 8, 32 // x element size and bytes per step of x
	if in32 {
		size, xstep = 4, 16
	}
	b.Raw("MOVQ CX, R11").Raw("SHRQ $4, R11").Raw("JZ done")
	b.Raw("MOVQ CX, R12").Raw("SHRQ $1, R12") // m
	b.Raw("MOVQ CX, R13").Raw("SHRQ $2, R13") // h
	b.Raw("LEAQ (R12)(R13*1), R14")           // m+h
	b.Raw("LEAQ (SI)(R12*%d), R8", size).Raw("SUBQ $%d, R8", xstep)
	b.Raw("LEAQ (SI)(R14*%d), R9", size)
	b.Raw("LEAQ -%d(R9), R10", xstep)
	b.Raw("LEAQ (DI)(R13*8), DX")
	b.Raw("VPCMPEQD Y7, Y7, Y7").Raw("VPSLLQ $63, Y7, Y7")
	b.Label("loop")
	if in32 {
		b.Raw("VMOVUPS (SI), X1").Raw("VCVTPS2PD X1, Y1")
		b.Raw("VMOVUPS (R8), X2").Raw("VPERMILPS $0x1B, X2, X2").Raw("VCVTPS2PD X2, Y2")
		b.Raw("VMOVUPS (R9), X4").Raw("VCVTPS2PD X4, Y4")
		b.Raw("VMOVUPS (R10), X5").Raw("VPERMILPS $0x1B, X5, X5").Raw("VCVTPS2PD X5, Y5")
	} else {
		b.Raw("VMOVUPD (SI), Y1")
		b.Raw("VMOVUPD (R8), Y2").Raw("VPERMPD $0x1B, Y2, Y2")
		b.Raw("VMOVUPD (R9), Y4")
		b.Raw("VMOVUPD (R10), Y5").Raw("VPERMPD $0x1B, Y5, Y5")
	}
	b.Raw("VSUBPD Y2, Y1, Y3") // C - D
	b.Raw("VADDPD Y5, Y4, Y6") // A + B
	b.Raw("VXORPD Y7, Y6, Y6") // -(B + A)
	b.Raw("VMOVUPD Y3, (DX)").Raw("VMOVUPD Y6, (DI)")
	b.Raw("ADDQ $%d, SI", xstep).Raw("ADDQ $%d, R9", xstep).Raw("SUBQ $%d, R8", xstep).Raw("SUBQ $%d, R10", xstep)
	b.Raw("ADDQ $32, DI").Raw("ADDQ $32, DX")
	b.Raw("DECQ R11").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}

// unfold: func(dst, u []float64), dst []float32 when out32.
func unfold(name string, out32 bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("u")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// SI S front, R8 T front, DI dst[i], R9 dst[m+h+i], R10 dst[m-i-4], R11 dst[m+h-i-4], R12 steps
	b.LoadArg("dst_base", "DI").LoadArg("u_base", "SI").LoadArg("u_len", "CX")
	size, dstep := 8, 32 // destination element size and bytes per step
	if out32 {
		size, dstep = 4, 16
	}
	b.Raw("MOVQ CX, R12").Raw("SHRQ $3, R12").Raw("JZ done")
	b.Raw("MOVQ CX, R13").Raw("SHRQ $1, R13") // h
	b.Raw("LEAQ (SI)(R13*8), R8")
	b.Raw("LEAQ (CX)(R13*1), R14") // m+h
	b.Raw("LEAQ (DI)(R14*%d), R9", size)
	b.Raw("LEAQ (DI)(CX*%d), R10", size).Raw("SUBQ $%d, R10", dstep)
	b.Raw("MOVQ R9, R11").Raw("SUBQ $%d, R11", dstep)
	b.Raw("VPCMPEQD Y7, Y7, Y7").Raw("VPSLLQ $63, Y7, Y7")
	b.Label("loop")
	b.Raw("VMOVUPD (SI), Y1").Raw("VMOVUPD (R8), Y2")
	b.Raw("VXORPD Y7, Y1, Y3").Raw("VXORPD Y7, Y2, Y4")         // -S, -T
	b.Raw("VPERMPD $0x1B, Y4, Y5").Raw("VPERMPD $0x1B, Y3, Y6") // reverse(-T), reverse(-S)
	if out32 {
		b.Raw("VCVTPD2PSY Y2, X2").Raw("VCVTPD2PSY Y3, X3").Raw("VCVTPD2PSY Y5, X5").Raw("VCVTPD2PSY Y6, X6")
		b.Raw("VMOVUPS X2, (DI)").Raw("VMOVUPS X3, (R9)").Raw("VMOVUPS X5, (R10)").Raw("VMOVUPS X6, (R11)")
	} else {
		b.Raw("VMOVUPD Y2, (DI)").Raw("VMOVUPD Y3, (R9)").Raw("VMOVUPD Y5, (R10)").Raw("VMOVUPD Y6, (R11)")
	}
	b.Raw("ADDQ $32, SI").Raw("ADDQ $32, R8")
	b.Raw("ADDQ $%d, DI", dstep).Raw("ADDQ $%d, R9", dstep).Raw("SUBQ $%d, R10", dstep).Raw("SUBQ $%d, R11", dstep)
	b.Raw("DECQ R12").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
