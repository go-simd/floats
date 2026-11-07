//go:build ignore

// Command gen produces extrema_arm64.s with go-asmgen: the NEON float64 Max and
// Min kernels maxNEON/minNEON (go-ndarray's, moved here: ndarray v0.37.0 had
// switched to floats.Max/Min, which were a scalar loop), for n >= 8: four D2
// accumulators seeded with the first eight elements, folded together, then the
// two lanes, then the tail one at a time. FMAX/FMIN return NaN when either
// operand is NaN and order -0 below +0, as the scalar FMAXD/FMIND that Go's
// builtin max/min compile to do, and the operation is associative under those
// rules, so the result is bit-identical to the loop (and to NumPy's max/min).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(extremumKernel("maxNEON", "VFMAX", "FMAXD"))
	f.Add(extremumKernel("minNEON", "VFMIN", "FMIND"))
	if err := os.WriteFile("extrema_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema_arm64.s")
}

func extremumKernel(name, vop, sop string) *emit.Function {
	sig := arm64.Layout([]string{"a", "n"}, []arm64.Type{arm64.Ptr, arm64.Int64},
		[]string{"ret"}, []arm64.Type{arm64.Float64})
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R0").LoadArg("n", "R1").
		Raw("VLD1.P 64(R0), [V0.D2, V1.D2, V2.D2, V3.D2]").
		Raw("SUB $8, R1").
		Raw("block:").Raw("CMP $8, R1").Raw("BLT fold").
		Raw("VLD1.P 64(R0), [V4.D2, V5.D2, V6.D2, V7.D2]")
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("%s V%d.D2, V%d.D2, V%d.D2", vop, 4+r, r, r))
	}
	b.Raw("SUB $8, R1").Raw("B block").
		Raw("fold:").
		Raw(vop+" V1.D2, V0.D2, V0.D2").
		Raw(vop+" V3.D2, V2.D2, V2.D2").
		Raw(vop+" V2.D2, V0.D2, V0.D2").
		Raw("VMOV V0.D[0], R3").
		Raw("VMOV V0.D[1], R4").
		Raw("FMOVD R3, F0").
		Raw("FMOVD R4, F9").
		Raw(sop+" F9, F0, F0").
		Raw("tail:").Raw("CBZ R1, done").
		Raw("FMOVD.P 8(R0), F10").
		Raw(sop+" F10, F0, F0").
		Raw("SUB $1, R1").Raw("B tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
