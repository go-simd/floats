//go:build ignore

// Command gen produces sqmag_arm64.s with go-asmgen: the NEON kernel of the squared
// magnitude (sqmag.go), two bins per step:
//
//	VLD1 c[2k : 2k+4]                 -> (re0 im0), (re1 im1)
//	VUZP1 / VUZP2                     -> Re = (re0 re1), Im = (im0 im1)
//	P = Re*Re ; P += Im*Im (VFMLA, one rounding fewer than the portable loop)
//	dst[k : k+2] = P * s
//
// The kernel does floor(n/2) steps and the Go caller finishes the last bin. Operand
// order: Go's arm64 assembler writes three-operand vector forms as `OP Vm, Vn, Vd`
// meaning Vd = Vn op Vm; VFMLA accumulates into Vd.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> sqmag_arm64_gen.go
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
	f.Add(kernel("sqmagNEON").Func())
	if err := os.WriteFile("sqmag_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sqmag_arm64.s")
}

func kernel(name string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("c"), abi.Scalar("s", abi.Float64)}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R2 dst, R0 c, R1 n, R14 steps
	b.LoadArg("dst_base", "R2").LoadArg("c_base", "R0").LoadArg("dst_len", "R1").LoadArg("s", "F13")
	b.Raw("VDUP V13.D[0], V13.D2")
	b.Raw("LSR $1, R1, R14").Raw("CBZ R14, done")
	b.Label("loop")
	b.Raw("VLD1.P 32(R0), [V1.D2, V2.D2]")
	b.Raw("VUZP1 V2.D2, V1.D2, V3.D2").Raw("VUZP2 V2.D2, V1.D2, V4.D2") // Re, Im
	b.Raw("VFMUL V3.D2, V3.D2, V5.D2").Raw("VFMLA V4.D2, V4.D2, V5.D2")
	b.Raw("VFMUL V13.D2, V5.D2, V5.D2")
	b.Raw("VST1.P [V5.D2], 16(R2)")
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}
