//go:build ignore

// Command gen produces cmulto_arm64.s with go-asmgen: the arm64 kernel of CMulTo
// (cmulto.go), n complex numbers in one call: steps of four, then the last n mod 4 one
// at a time. It is the scalar loop, not NEON: a complex128 is as wide as a 128-bit
// vector, so NEON saves no multiply and pays shuffles (LD2/ST2 or ZIP, DUP, EXT) on the
// one or two vector pipes that do the multiplies; on Neoverse-N1 and -N2 the NEON kernel
// of v0.32.0 lost to Go's loop below n = 64 (go-simd/floats#81). This loop is Go's,
// without its bounds checks, its call of a tail, or its loop overhead, and measured
// faster than both at every n (see the pull request).
//
// One complex number: FLDPD loads (ar, ai) and (br, bi); E = ar*br (FMULD), E -= ai*bi
// (FMSUBD) is the real part; F = ar*bi (FMULD), F += ai*br (FMADDD) the imaginary one;
// FSTPD stores them. Reads of a number are done before its write, so dst may be a or b
// exactly.
//
// Operand order: `FMULD Fm, Fn, Fd` is Fd = Fn*Fm; `FMADDD Fm, Fa, Fn, Fd` is
// Fd = Fa + Fn*Fm; `FMSUBD Fm, Fa, Fn, Fd` is Fd = Fa - Fn*Fm.
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

// one emits the product of one complex number, four registers of inputs from base fb.
func one(b *arm64.Builder, fb int) {
	ar, ai, br, bi, re, im := fb, fb+1, fb+2, fb+3, fb+4, fb+5
	b.Raw(fmt.Sprintf("FLDPD.P 16(R1), (F%d, F%d)", ar, ai))
	b.Raw(fmt.Sprintf("FLDPD.P 16(R2), (F%d, F%d)", br, bi))
	b.Raw(fmt.Sprintf("FMULD F%d, F%d, F%d", br, ar, re))
	b.Raw(fmt.Sprintf("FMSUBD F%d, F%d, F%d, F%d", bi, re, ai, re))
	b.Raw(fmt.Sprintf("FMULD F%d, F%d, F%d", bi, ar, im))
	b.Raw(fmt.Sprintf("FMADDD F%d, F%d, F%d, F%d", br, im, ai, im))
	b.Raw(fmt.Sprintf("FSTPD.P (F%d, F%d), 16(R0)", re, im))
}

func kernel() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Scalar("dst", abi.Ptr), abi.Scalar("a", abi.Ptr), abi.Scalar("b", abi.Ptr), abi.Scalar("n", abi.Int64)}, nil)
	b := arm64.NewFunc("cmulToFP", sig, 0)
	// R0 dst, R1 a, R2 b, R4 n complex numbers, R5 steps of four.
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("n", "R4").
		Raw("LSR $2, R4, R5").Raw("CBZ R5, rem").Label("loop")
	for k := 0; k < 4; k++ {
		one(b, 8*k)
	}
	b.Raw("SUBS $1, R5, R5").Raw("BNE loop").Label("rem").
		Raw("AND $3, R4, R4").Raw("CBZ R4, done").Label("remloop")
	one(b, 0)
	b.Raw("SUBS $1, R4, R4").Raw("BNE remloop").Label("done").Ret()
	return b
}
