//go:build ignore

// Command gen produces conjrev_amd64.s with go-asmgen: the AVX2 kernel of the
// conjugate-reverse (conjrev.go), four complex numbers per step. With n = len(dst)/2,
// S the source address of the last four numbers (src + 16(n-4)) moving down and D the
// destination moving up:
//
//	Y0 = S[0:2] numbers (n-4-k, n-3-k)    Y1 = S[2:4] numbers (n-2-k, n-1-k)
//	VPERMPD $0x4E swaps the 128-bit halves of each: the pair reversed
//	VXORPD with the sign of the imaginary lanes (VBLENDPD of the sign bit and zero)
//	dst[k : k+4] = (Y1 reversed, Y0 reversed)
//
// Only moves and sign flips: the result is the portable loop's exactly. The kernel does
// floor(n/4) steps and the Go caller finishes the rest. Go's amd64 assembler writes AVX
// forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> conjrev_amd64_gen.go
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
	f.Add(kernel("conjRevAVX2").Func())
	if err := os.WriteFile("conjrev_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote conjrev_amd64.s")
}

func kernel(name string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, SI src (moving down), CX floats then complex count, R8 steps, Y7 sign mask
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("dst_len", "CX")
	b.Raw("SHRQ $1, CX")
	b.Raw("MOVQ CX, R8").Raw("SHRQ $2, R8").Raw("JZ done")
	b.Raw("MOVQ CX, R9").Raw("SHLQ $4, R9").Raw("ADDQ R9, SI").Raw("SUBQ $64, SI") // src + 16(n-4)
	b.Raw("VPCMPEQD Y5, Y5, Y5").Raw("VPSLLQ $63, Y5, Y5").Raw("VXORPD Y6, Y6, Y6")
	b.Raw("VBLENDPD $0x0A, Y5, Y6, Y7") // sign bit in the imaginary lanes 1 and 3
	b.Label("loop")
	b.Raw("VMOVUPD (SI), Y0").Raw("VMOVUPD 32(SI), Y1").Raw("SUBQ $64, SI")
	b.Raw("VPERMPD $0x4E, Y1, Y1").Raw("VPERMPD $0x4E, Y0, Y0")
	b.Raw("VXORPD Y7, Y1, Y1").Raw("VXORPD Y7, Y0, Y0")
	b.Raw("VMOVUPD Y1, (DI)").Raw("VMOVUPD Y0, 32(DI)").Raw("ADDQ $64, DI")
	b.Raw("DECQ R8").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
