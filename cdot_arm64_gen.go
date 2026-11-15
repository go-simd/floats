//go:build ignore

// Command gen produces cdot_arm64.s with go-asmgen: the NEON kernel of CDotC (cdot.go).
// Per block of four complex numbers (8 float64 of each input): V8..V11 = a, V12..V15 = b,
// V16..V19 = b with its two halves swapped (VEXT $8: (bi, br)); V0..V3 += a*b (lanes
// ar*br, ai*bi) and V4..V7 += a*swap(b) (ar*bi, ai*br), one VFMLA each. After the loop the
// four registers of each sum are added and stored: out[0:2] = V0, out[2:4] = V4. R0 = a,
// R2 = b, R3 = out, R4 = blocks.
//
// Operand order: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> cdot_arm64_gen.go
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
	if err := os.WriteFile("cdot_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cdot_arm64.s")
}

func kernel() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("a"), abi.Slice("b"), abi.Slice("out")}, nil)
	b := arm64.NewFunc("cdotcNEON", sig, 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R2").LoadArg("out_base", "R3").
		Raw("LSR $3, R1, R4")
	for k := 0; k < 8; k++ {
		b.Raw(fmt.Sprintf("VEOR V%d.B16, V%d.B16, V%d.B16", k, k, k))
	}
	b.Raw("CBZ R4, done").Label("loop").
		Raw("VLD1.P 64(R0), [V8.D2, V9.D2, V10.D2, V11.D2]").
		Raw("VLD1.P 64(R2), [V12.D2, V13.D2, V14.D2, V15.D2]")
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VEXT $8, V%d.B16, V%d.B16, V%d.B16", 12+k, 12+k, 16+k))
	}
	for k := 0; k < 4; k++ {
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", 12+k, 8+k, k))   // sum1 += a*b
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", 16+k, 8+k, 4+k)) // sum2 += a*swap(b)
	}
	b.Raw("SUBS $1, R4, R4").Raw("BNE loop").Label("done")
	for _, g := range []int{0, 4} {
		b.Raw(fmt.Sprintf("VFADD V%d.D2, V%d.D2, V%d.D2", g+1, g, g)).
			Raw(fmt.Sprintf("VFADD V%d.D2, V%d.D2, V%d.D2", g+3, g+2, g+2)).
			Raw(fmt.Sprintf("VFADD V%d.D2, V%d.D2, V%d.D2", g+2, g, g))
	}
	b.Raw("VST1 [V0.D2], (R3)").Raw("ADD $16, R3, R3").Raw("VST1 [V4.D2], (R3)").Ret()
	return b
}
