//go:build ignore

// Command gen produces cmulto_amd64.s with go-asmgen: the AVX2+FMA kernel of CMulTo
// (cmulto.go), four complex numbers per step, the floor(n/4) steps of which the Go
// caller follows with the last numbers. For c = (ar, ai) pairs of a and w = (br, bi) of b:
// Y7 = (br, br) (VMOVDDUP), Y8 = (bi, bi) (VPERMILPD $15), Y9 = swap(c)*(bi, bi) =
// (ai*bi, ar*bi), then VFMADDSUB231PD c, Y7, Y9 = c*(br, br) -/+ Y9 = (ar*br - ai*bi,
// ai*br + ar*bi): one multiply and one fused multiply-add per two-number register.
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
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("steps", abi.Int64)}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, SI a, DX b, CX steps of 4 complex numbers (8 float64).
	b.LoadArg("dst", "DI").LoadArg("a", "SI").LoadArg("b", "DX").LoadArg("steps", "CX")
	b.Raw("TESTQ CX, CX").Raw("JZ done")
	b.Label("loop")
	for i := 0; i < 2; i++ {
		off := 32 * i
		c, w := fmt.Sprintf("Y%d", i), fmt.Sprintf("Y%d", 2+i)
		b.Raw(fmt.Sprintf("VMOVUPD %d(SI), %s", off, c))
		b.Raw(fmt.Sprintf("VMOVUPD %d(DX), %s", off, w))
	}
	for i := 0; i < 2; i++ {
		c, w, t := fmt.Sprintf("Y%d", i), fmt.Sprintf("Y%d", 2+i), fmt.Sprintf("Y%d", 4+i)
		b.Raw(fmt.Sprintf("VMOVDDUP %s, %s", w, fmt.Sprintf("Y%d", 6+i))) // (br, br)
		b.Raw(fmt.Sprintf("VPERMILPD $15, %s, %s", w, w))                 // (bi, bi)
		b.Raw(fmt.Sprintf("VPERMILPD $5, %s, %s", c, t))                  // swap(c)
		b.Raw(fmt.Sprintf("VMULPD %s, %s, %s", w, t, t))                  // swap(c)*(bi, bi)
		b.Raw(fmt.Sprintf("VFMADDSUB231PD %s, Y%d, %s", c, 6+i, t))       // c*(br, br) -/+ t
	}
	for i := 0; i < 2; i++ {
		b.Raw(fmt.Sprintf("VMOVUPD Y%d, %d(DI)", 4+i, 32*i))
	}
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").Raw("ADDQ $64, DI")
	b.Raw("DECQ CX").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
