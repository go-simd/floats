//go:build ignore

// Command gen produces cmulto_amd64.s with go-asmgen: the AVX2+FMA kernel of CMulTo
// (cmulto.go), n complex numbers in one call: steps of four, then a step of two and a
// step of one when n has those bits, so the Go caller has no tail to call (go-simd/
// floats#82: the call of a tail and a second dispatch cost more than the kernel at the
// sizes of a small transform). For c = (ar, ai) pairs of a and w = (br, bi) of b:
// Y6 = (br, br) (VMOVDDUP), Y2 = (bi, bi) (VPERMILPD $15), Y4 = swap(c)*(bi, bi) =
// (ai*bi, ar*bi), then VFMADDSUB231PD c, Y6, Y4 = c*(br, br) -/+ Y4 = (ar*br - ai*bi,
// ai*br + ar*bi): one multiply and one fused multiply-add per two-number register.
// The operands of b are memory operands of the shuffles, which saves the loads.
// Reads are done before the writes of the same step, so dst may be a or b exactly.
// Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen> cmulto_amd64_gen.go
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
	f.Add(kernel("cmulToAVX2").Func())
	if err := os.WriteFile("cmulto_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmulto_amd64.s")
}

func kernel(name string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("n", abi.Int64)}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, SI a, DX b, AX n complex numbers, CX steps of four.
	b.LoadArg("dst", "DI").LoadArg("a", "SI").LoadArg("b", "DX").LoadArg("n", "AX")
	b.Raw("MOVQ AX, CX").Raw("SHRQ $2, CX").Raw("JZ two")
	b.Label("loop")
	b.Raw("VMOVUPD (SI), Y0").Raw("VMOVUPD 32(SI), Y1")
	b.Raw("VMOVDDUP (DX), Y6").Raw("VMOVDDUP 32(DX), Y7")
	b.Raw("VPERMILPD $15, (DX), Y2").Raw("VPERMILPD $15, 32(DX), Y3")
	b.Raw("VPERMILPD $5, Y0, Y4").Raw("VPERMILPD $5, Y1, Y5")
	b.Raw("VMULPD Y2, Y4, Y4").Raw("VMULPD Y3, Y5, Y5")
	b.Raw("VFMADDSUB231PD Y0, Y6, Y4").Raw("VFMADDSUB231PD Y1, Y7, Y5")
	b.Raw("VMOVUPD Y4, (DI)").Raw("VMOVUPD Y5, 32(DI)")
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").Raw("ADDQ $64, DI")
	b.Raw("DECQ CX").Raw("JNZ loop")
	b.Label("two").Raw("TESTQ $2, AX").Raw("JZ one")
	b.Raw("VMOVUPD (SI), Y0").Raw("VMOVDDUP (DX), Y6").Raw("VPERMILPD $15, (DX), Y2")
	b.Raw("VPERMILPD $5, Y0, Y4").Raw("VMULPD Y2, Y4, Y4").Raw("VFMADDSUB231PD Y0, Y6, Y4")
	b.Raw("VMOVUPD Y4, (DI)")
	b.Raw("ADDQ $32, SI").Raw("ADDQ $32, DX").Raw("ADDQ $32, DI")
	b.Label("one").Raw("TESTQ $1, AX").Raw("JZ done")
	b.Raw("VMOVUPD (SI), X0").Raw("VMOVDDUP (DX), X6").Raw("VPERMILPD $3, (DX), X2")
	b.Raw("VPERMILPD $1, X0, X4").Raw("VMULPD X2, X4, X4").Raw("VFMADDSUB231PD X0, X6, X4")
	b.Raw("VMOVUPD X4, (DI)")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
