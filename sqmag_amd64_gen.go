//go:build ignore

// Command gen produces sqmag_amd64.s with go-asmgen: the AVX2+FMA kernel of the
// squared magnitude (sqmag.go), four bins per step:
//
//	Y0 = c[2k : 2k+4], Y1 = c[2k+4 : 2k+8]
//	Re = VSHUFPD $0 then VPERMPD $0xD8   (re0 re1 re2 re3)
//	Im = VSHUFPD $0xF then VPERMPD $0xD8
//	P = Re*Re ; P += Im*Im (VFMADD231PD, one rounding fewer than the portable loop)
//	dst[k : k+4] = P * s
//
// The kernel does floor(n/4) steps and the Go caller finishes the rest; the gate is
// hasAVX2FMA like the other AVX2 kernels. Go's amd64 assembler writes AVX forms as
// `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> sqmag_amd64_gen.go
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
	f.Add(kernel("sqmagAVX2").Func())
	if err := os.WriteFile("sqmag_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sqmag_amd64.s")
}

func kernel(name string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("c"), abi.Scalar("s", abi.Float64)}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, SI c, CX n, R8 steps, Y7 = s
	b.LoadArg("dst_base", "DI").LoadArg("c_base", "SI").LoadArg("dst_len", "CX").LoadArg("s", "X7")
	b.Raw("VBROADCASTSD X7, Y7")
	b.Raw("MOVQ CX, R8").Raw("SHRQ $2, R8").Raw("JZ done")
	b.Label("loop")
	b.Raw("VMOVUPD (SI), Y0").Raw("VMOVUPD 32(SI), Y1").Raw("ADDQ $64, SI")
	b.Raw("VSHUFPD $0x00, Y1, Y0, Y2").Raw("VPERMPD $0xD8, Y2, Y2") // Re
	b.Raw("VSHUFPD $0x0F, Y1, Y0, Y3").Raw("VPERMPD $0xD8, Y3, Y3") // Im
	b.Raw("VMULPD Y2, Y2, Y4").Raw("VFMADD231PD Y3, Y3, Y4")
	b.Raw("VMULPD Y7, Y4, Y4")
	b.Raw("VMOVUPD Y4, (DI)").Raw("ADDQ $32, DI")
	b.Raw("DECQ R8").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
