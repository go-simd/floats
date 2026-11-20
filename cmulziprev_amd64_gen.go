//go:build ignore

// Command gen produces cmulziprev_amd64.s with go-asmgen: the AVX2+FMA kernel of
// CMulZipRev (cmulziprev.go), four complex numbers per step. Y0 = x[j:j+4], Y1 = the
// four y read from the far end and reversed (VPERMPD $0x1B); VUNPCKLPD/VUNPCKHPD pair
// them (x0,y0',x2,y2') and (x1,y1',x3,y3'), VPERM2F128 puts the numbers in order, a
// sign flip of the imaginary lanes makes x - i y', and each product with w is
// VFMADDSUB231PD of c*(wr,wr) onto (ci,cr)*(wi,wi). The kernel does floor(m/4) steps and
// the Go caller finishes the rest. Go's amd64 assembler writes AVX forms as
// `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen> cmulziprev_amd64_gen.go
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
	f.Add(kernel("cmulZipRevAVX2").Func())
	if err := os.WriteFile("cmulziprev_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmulziprev_amd64.s")
}

func kernel(name string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x"), abi.Slice("y"), abi.Slice("w")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, R8 x moving up, R9 y moving down, R10 w, CX floats then steps.
	b.LoadArg("dst_base", "DI").LoadArg("x_base", "R8").LoadArg("y_base", "R9").LoadArg("w_base", "R10").LoadArg("dst_len", "CX")
	b.Raw("MOVQ CX, R12").Raw("SHLQ $2, R12").Raw("ADDQ R12, R9").Raw("SUBQ $32, R9") // y + 8(m-4)
	b.Raw("SHRQ $3, CX").Raw("JZ done")                                               // m/4 steps
	b.Raw("VPCMPEQD Y13, Y13, Y13").Raw("VPSLLQ $63, Y13, Y13").Raw("VXORPD Y12, Y12, Y12")
	b.Raw("VBLENDPD $0x0A, Y13, Y12, Y14")
	b.Label("loop")
	b.Raw("VMOVUPD (R8), Y0").Raw("VMOVUPD (R9), Y1").Raw("VPERMPD $0x1B, Y1, Y1")
	b.Raw("VUNPCKLPD Y1, Y0, Y2").Raw("VUNPCKHPD Y1, Y0, Y3")
	b.Raw("VPERM2F128 $0x20, Y3, Y2, Y4").Raw("VPERM2F128 $0x31, Y3, Y2, Y5")
	b.Raw("VXORPD Y14, Y4, Y4").Raw("VXORPD Y14, Y5, Y5")
	for i, c := range []string{"Y4", "Y5"} {
		off := 32 * i
		b.Raw(fmt.Sprintf("VMOVUPD %d(R10), Y6", off)).Raw("VMOVDDUP Y6, Y7").Raw("VPERMILPD $15, Y6, Y8")
		b.Raw(fmt.Sprintf("VPERMILPD $5, %s, Y9", c)).Raw("VMULPD Y8, Y9, Y9").Raw(fmt.Sprintf("VFMADDSUB231PD %s, Y7, Y9", c))
		b.Raw(fmt.Sprintf("VMOVUPD Y9, %d(DI)", off))
	}
	b.Raw("ADDQ $32, R8").Raw("SUBQ $32, R9").Raw("ADDQ $64, R10").Raw("ADDQ $64, DI")
	b.Raw("DECQ CX").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
