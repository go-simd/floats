//go:build ignore

// Command gen produces extrema32_arm64.s with go-asmgen: the NEON float32 Max
// and Min kernels max32NEON/min32NEON (go-ndarray's, moved here so the kernel
// lives once), for n >= 16: four .S4 accumulators seeded with the first 16
// elements, folded together, then the four lanes, then the tail one at a time.
// VFMAX/VFMIN, like the scalar FMAXS/FMINS that Go's builtin max/min compile to,
// return NaN when either operand is NaN and order -0 below +0, and the operation
// is associative under those rules, so the result is the sequential loop's bit
// for bit (and NumPy's max/min).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(extremum32Kernel("max32NEON", "VFMAX", "FMAXS"))
	f.Add(extremum32Kernel("min32NEON", "VFMIN", "FMINS"))
	if err := os.WriteFile("extrema32_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_arm64.s")
}

func extremum32Kernel(name, vop, sop string) *emit.Function {
	sig := arm64.Layout([]string{"a", "n"}, []arm64.Type{arm64.Ptr, arm64.Int64},
		[]string{"ret"}, []arm64.Type{arm64.Float32})
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R0").LoadArg("n", "R1").
		Raw("VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]").
		Raw("SUB $16, R1").
		Raw("block:").Raw("CMP $16, R1").Raw("BLT fold").
		Raw("VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]")
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("%s V%d.S4, V%d.S4, V%d.S4", vop, 4+r, r, r))
	}
	b.Raw("SUB $16, R1").Raw("B block").
		Raw("fold:").
		Raw(vop + " V1.S4, V0.S4, V0.S4").
		Raw(vop + " V3.S4, V2.S4, V2.S4").
		Raw(vop + " V2.S4, V0.S4, V0.S4")
	for l := 0; l < 4; l++ {
		b.Raw(fmt.Sprintf("VMOV V0.S[%d], R%d", l, 3+l))
	}
	b.Raw("FMOVS R3, F0").Raw("FMOVS R4, F9").Raw("FMOVS R5, F10").Raw("FMOVS R6, F11").
		Raw(sop+" F9, F0, F0").Raw(sop+" F11, F10, F10").Raw(sop+" F10, F0, F0").
		Raw("tail:").Raw("CBZ R1, done").
		Raw("FMOVS.P 4(R0), F10").
		Raw(sop+" F10, F0, F0").
		Raw("SUB $1, R1").Raw("B tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
