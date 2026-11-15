//go:build ignore

// Command gen produces round_arm64.s with go-asmgen: the float64 elementwise Abs,
// Neg, Square, Floor, Ceil, Trunc and RoundEven kernels (go-ndarray/ndarray#118).
//
// Sources: Arm Architecture Reference Manual (DDI 0602, AdvSIMD floating-point
// instructions, 2D arrangement). The online page did not render when this was
// written, so the semantics below are those of the manual as known and were
// verified by execution on Apple silicon (round_test.go, bit for bit):
//
//   - FABS (vector) clears and FNEG (vector) flips the sign bit of each lane and
//     nothing else, so NaNs keep their payload; Go names them VFABS and VFNEG.
//   - FMUL (vector): one IEEE multiplication per lane (VFMUL), no fusion.
//   - FRINTM (round toward minus infinity) = Floor, FRINTP (toward plus infinity)
//     = Ceil, FRINTZ (toward zero) = Trunc, FRINTN (to nearest, ties to even) =
//     RoundEven. They are named VFRINTM, VFRINTP, VFRINTZ, VFRINTN. Unlike
//     FRINTX/FRINTI they use the rounding the mnemonic names, not FPCR.RMode,
//     and they never signal Inexact. A quiet NaN passes through unchanged, a
//     signalling NaN is quieted, zeros and infinities are unchanged.
//
// Go's arm64 assembler writes a two-operand vector form as `OP Vn, Vd`. Each
// kernel does n/8 blocks of eight elements (n a positive multiple of 8, sliced by
// the Go caller, which finishes the last seven elements); a block is loaded,
// computed and stored before the next, so dst may alias src exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_arm64_gen.go
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
	for _, k := range []struct{ name, op string }{
		{"absNEON", "VFABS"}, {"negNEON", "VFNEG"}, {"squareNEON", "VFMUL"},
		{"floorNEON", "VFRINTM"}, {"ceilNEON", "VFRINTP"},
		{"truncNEON", "VFRINTZ"}, {"rintNEON", "VFRINTN"},
	} {
		f.Add(kernel(k.name, k.op).Func())
	}
	if err := os.WriteFile("round_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_arm64.s")
}

func kernel(name, op string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("src_base", "R0").LoadArg("src_len", "R1").
		Raw("MOVD $0, R3").Label("loop").
		Raw("ADD R3<<3, R0, R5").
		Raw("VLD1 (R5), [V1.D2, V2.D2, V3.D2, V4.D2]")
	for k := 1; k <= 4; k++ {
		if op == "VFMUL" {
			b.Raw("VFMUL V%d.D2, V%d.D2, V%d.D2", k, k, k)
		} else {
			b.Raw("%s V%d.D2, V%d.D2", op, k, k)
		}
	}
	b.Raw("ADD R3<<3, R2, R6").
		Raw("VST1 [V1.D2, V2.D2, V3.D2, V4.D2], (R6)").
		Raw("ADD $8, R3, R3").Raw("CMP R1, R3").Raw("BLT loop").Ret()
	return b
}
