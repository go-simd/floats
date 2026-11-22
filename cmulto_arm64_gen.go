//go:build ignore

// Command gen produces cmulto_arm64.s with go-asmgen: the NEON kernel of CMulTo
// (cmulto.go), four complex numbers per step. VLD2 de-interleaves two complex numbers
// into (V0 = ar, V1 = ai) and (V2 = br, V3 = bi); V4 = ar*br - ai*bi (VFMUL, VFMLS),
// V5 = ar*bi + ai*br (VFMUL, VFMLA); VST2 interleaves them back. Reads of a step are
// done before its writes, so dst may be a or b exactly.
//
// Operand order: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> cmulto_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(kernel().Func())
	if err := os.WriteFile("cmulto_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmulto_arm64.s")
}

func kernel() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("steps", abi.Int64)}, nil)
	b := arm64.NewFunc("cmulToNEON", sig, 0)
	// R0 dst, R1 a, R2 b, R4 steps of 4 complex numbers.
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("steps", "R4").
		Raw("CBZ R4, done").Label("loop")
	for k := 0; k < 2; k++ {
		b.Raw(fmt.Sprintf("VLD2.P 32(R1), [V%d.D2, V%d.D2]", 8*k, 8*k+1))
		b.Raw(fmt.Sprintf("VLD2.P 32(R2), [V%d.D2, V%d.D2]", 8*k+2, 8*k+3))
	}
	for k := 0; k < 2; k++ {
		ar, ai, br, bi, re, im := 8*k, 8*k+1, 8*k+2, 8*k+3, 8*k+4, 8*k+5
		b.Raw(fmt.Sprintf("VFMUL V%d.D2, V%d.D2, V%d.D2", br, ar, re))
		b.Raw(fmt.Sprintf("VFMUL V%d.D2, V%d.D2, V%d.D2", bi, ar, im))
		b.Raw(fmt.Sprintf("VFMLS V%d.D2, V%d.D2, V%d.D2", bi, ai, re))
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", br, ai, im))
		b.Raw(fmt.Sprintf("VST2.P [V%d.D2, V%d.D2], 32(R0)", re, im))
	}
	b.Raw("SUBS $1, R4, R4").Raw("BNE loop").Label("done").Ret()
	return b
}
