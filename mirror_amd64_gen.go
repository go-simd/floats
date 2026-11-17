//go:build ignore

// Command gen produces mirror_amd64.s with go-asmgen: the AVX2+FMA kernels of the
// mirror combiner (mirror.go), two values of k per step. With a = z[k] and b = z[n-k]
// (the two numbers of the step loaded from the far end, their lanes swapped):
//
//	u = a + conj(b)           v = -i(a - conj(b))      r = w1[k]*u + w2[k]*v
//
// and a complex product x*w is VFMADDSUB231PD of x*(wr,wr) onto (xi,xr)*(wi,wi):
// even lanes subtract, odd lanes add. mirrorCombineAVX2 stores r as complex numbers;
// mirrorCombineSplitAVX2 stores the real parts at dst[k], dst[k+1] and the negated
// imaginary parts, mirrored, at dst[2n-k-1], dst[2n-k]. Each does floor((n-1)/2)
// steps (k = 1 ...) and the Go caller finishes the rest. Go's amd64 assembler writes AVX
// forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen> mirror_amd64_gen.go
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
	f.Add(kernel("mirrorCombineAVX2", false).Func())
	f.Add(kernel("mirrorCombineSplitAVX2", true).Func())
	if err := os.WriteFile("mirror_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mirror_amd64.s")
}

func kernel(name string, split bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("z"), abi.Slice("w1"), abi.Slice("w2")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst (low), SI dst (high, split only), R8 a moving up, R13 b moving down,
	// R10 w1, R11 w2, CX complex count then steps. Y13 all signs, Y14 odd-lane signs.
	b.LoadArg("dst_base", "DI").LoadArg("z_base", "R8").LoadArg("z_len", "CX")
	b.LoadArg("w1_base", "R10").LoadArg("w2_base", "R11")
	b.Raw("SHRQ $1, CX")                                                         // n
	b.Raw("MOVQ CX, R12").Raw("SUBQ $1, R12").Raw("SHRQ $1, R12").Raw("JZ done") // steps
	b.Raw("MOVQ CX, R9").Raw("SHLQ $4, R9")                                      // 16n
	b.Raw("MOVQ R8, R13").Raw("ADDQ R9, R13").Raw("SUBQ $32, R13")               // z + 16(n-2)
	b.Raw("ADDQ $16, R8").Raw("ADDQ $16, R10").Raw("ADDQ $16, R11")              // k = 1
	if split {
		b.Raw("MOVQ DI, SI").Raw("ADDQ R9, SI").Raw("SUBQ $16, SI") // dst + 8(2n-2)
		b.Raw("ADDQ $8, DI")                                        // dst[1]
	} else {
		b.Raw("ADDQ $16, DI")
	}
	b.Raw("VPCMPEQD Y13, Y13, Y13").Raw("VPSLLQ $63, Y13, Y13").Raw("VXORPD Y12, Y12, Y12")
	b.Raw("VBLENDPD $0x0A, Y13, Y12, Y14")
	b.Label("loop")
	b.Raw("VMOVUPD (R8), Y0").Raw("VMOVUPD (R13), Y1").Raw("VPERM2F128 $1, Y1, Y1, Y1")
	b.Raw("VXORPD Y14, Y1, Y1")
	b.Raw("VADDPD Y1, Y0, Y2").Raw("VSUBPD Y1, Y0, Y3")
	b.Raw("VPERMILPD $5, Y3, Y3").Raw("VXORPD Y14, Y3, Y3")
	b.Raw("VMOVUPD (R10), Y4").Raw("VMOVDDUP Y4, Y5").Raw("VPERMILPD $15, Y4, Y6")
	b.Raw("VPERMILPD $5, Y2, Y7").Raw("VMULPD Y6, Y7, Y7").Raw("VFMADDSUB231PD Y2, Y5, Y7")
	b.Raw("VMOVUPD (R11), Y4").Raw("VMOVDDUP Y4, Y5").Raw("VPERMILPD $15, Y4, Y6")
	b.Raw("VPERMILPD $5, Y3, Y9").Raw("VMULPD Y6, Y9, Y9").Raw("VFMADDSUB231PD Y3, Y5, Y9")
	b.Raw("VADDPD Y9, Y7, Y7")
	if split {
		b.Raw("VPERMPD $0x08, Y7, Y8").Raw("VMOVUPD X8, (DI)")
		b.Raw("VPERMPD $0x07, Y7, Y8").Raw("VXORPD Y13, Y8, Y8").Raw("VMOVUPD X8, (SI)")
		b.Raw("ADDQ $16, DI").Raw("SUBQ $16, SI")
	} else {
		b.Raw("VMOVUPD Y7, (DI)").Raw("ADDQ $32, DI")
	}
	b.Raw("ADDQ $32, R8").Raw("SUBQ $32, R13").Raw("ADDQ $32, R10").Raw("ADDQ $32, R11")
	b.Raw("DECQ R12").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
