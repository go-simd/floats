//go:build ignore

// Command gen produces conjrev_arm64.s with go-asmgen: the NEON kernel of the
// conjugate-reverse (conjrev.go), four complex numbers per step. With n = len(dst)/2,
// S the source address of the last four numbers (src + 16(n-4)) moving down:
//
//	V1..V4 = the numbers n-4-k .. n-1-k      (VLD1, four registers)
//	V5..V8 = V4..V1 with the imaginary lane negated: VEOR with (0, sign bit)
//	dst[k : k+4] = V5..V8                    (VST1.P, four registers)
//
// Only moves and sign flips: the result is the portable loop's exactly. The kernel does
// floor(n/4) steps and the Go caller finishes the rest.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> conjrev_arm64_gen.go
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
	f.Add(kernel("conjRevNEON").Func())
	if err := os.WriteFile("conjrev_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote conjrev_arm64.s")
}

func kernel(name string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R2 dst, R0 src (moving down), R1 floats then complex count, R14 steps, V13 sign mask
	b.LoadArg("dst_base", "R2").LoadArg("src_base", "R0").LoadArg("dst_len", "R1")
	b.Raw("LSR $1, R1, R1")
	b.Raw("LSR $2, R1, R14").Raw("CBZ R14, done")
	b.Raw("LSL $4, R1, R3").Raw("ADD R3, R0, R0").Raw("SUB $64, R0, R0") // src + 16(n-4)
	b.Raw("MOVD $0x8000000000000000, R4")
	b.Raw("VEOR V13.B16, V13.B16, V13.B16").Raw("VMOV R4, V13.D[1]") // (0, sign bit)
	b.Label("loop")
	b.Raw("VLD1 (R0), [V1.D2, V2.D2, V3.D2, V4.D2]").Raw("SUB $64, R0, R0")
	b.Raw("VEOR V13.B16, V4.B16, V5.B16").Raw("VEOR V13.B16, V3.B16, V6.B16")
	b.Raw("VEOR V13.B16, V2.B16, V7.B16").Raw("VEOR V13.B16, V1.B16, V8.B16")
	b.Raw("VST1.P [V5.D2, V6.D2, V7.D2, V8.D2], 64(R2)")
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}
