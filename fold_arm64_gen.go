//go:build ignore

// Command gen produces fold_arm64.s with go-asmgen: the NEON kernels of the MDCT
// fold and unfold (fold.go), two lanes of float64 per step.
//
// Fold, for n = len(x), m = n/2, h = n/4 and the block at i (i, i+1):
//
//	C = x[i : i+2]                    D = reverse(x[m-i-2 : m-i])
//	A = x[m+h+i : m+h+i+2]            B = reverse(x[m+h-i-2 : m+h-i])
//	dst[h+i : h+i+2] = C - D          dst[i : i+2] = -(B + A)
//
// Unfold, for m = len(u), h = m/2 and the block at i:
//
//	S = u[i : i+2]                    T = u[h+i : h+i+2]
//	dst[i : i+2] = T                  dst[m-i-2 : m-i] = reverse(-T)
//	dst[m+h+i : m+h+i+2] = -S         dst[m+h-i-2 : m+h-i] = reverse(-S)
//
// The reversal of a pair of float64 is VEXT #8, and for a pair of float32 VREV64.
// The float32 forms widen the input with VFCVTL (the sum is float64) and narrow the
// output with VFCVTN. Nothing is fused: every output is one addition, subtraction or
// negation, the portable loop's result exactly. The kernels do the blocks that fit,
// floor(h/2) steps; the Go caller finishes the tail. Front streams advance with
// post-incremented loads and stores, back streams by one SUB.
//
// Operand order: Go's arm64 assembler writes three-operand vector forms as
// `OP Vm, Vn, Vd` meaning Vd = Vn op Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> fold_arm64_gen.go
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
	f.Add(fold("mdctFold64NEON", false).Func())
	f.Add(fold("mdctFoldFrom32NEON", true).Func())
	f.Add(unfold("mdctUnfold64NEON", false).Func())
	f.Add(unfold("mdctUnfoldTo32NEON", true).Func())
	if err := os.WriteFile("fold_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote fold_arm64.s")
}

// fold: func(dst, x []float64), x []float32 when in32.
func fold(name string, in32 bool) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R6 C front, R8 D back, R9 A front, R10 B back, R2 dst[i], R7 dst[h+i], R14 steps
	b.LoadArg("dst_base", "R2").LoadArg("x_base", "R0").LoadArg("x_len", "R1")
	sh, step := 3, 16 // element shift and bytes per step of x
	if in32 {
		sh, step = 2, 8
	}
	b.Raw("LSR $3, R1, R14").Raw("CBZ R14, done")
	b.Raw(fmt.Sprintf("LSL $%d, R1, R3", sh-1)) // m bytes
	if sh == 3 {
		b.Raw("LSL $1, R1, R4") // h bytes
	} else {
		b.Raw("MOVD R1, R4")
	}
	b.Raw("ADD R3, R4, R5") // (m+h) bytes
	b.Raw("MOVD R0, R6")
	b.Raw(fmt.Sprintf("ADD R3, R0, R8")).Raw(fmt.Sprintf("SUB $%d, R8, R8", step))
	b.Raw("ADD R5, R0, R9")
	b.Raw(fmt.Sprintf("SUB $%d, R9, R10", step))
	b.Raw("LSL $1, R1, R7").Raw("ADD R2, R7, R7") // dst + h*8
	b.Label("loop")
	if in32 {
		b.Raw("VLD1.P 8(R6), [V1.S2]").Raw("VFCVTL V1.S2, V1.D2")
		b.Raw("VLD1 (R8), [V2.S2]").Raw("SUB $8, R8, R8").Raw("VREV64 V2.S2, V2.S2").Raw("VFCVTL V2.S2, V2.D2")
		b.Raw("VLD1.P 8(R9), [V4.S2]").Raw("VFCVTL V4.S2, V4.D2")
		b.Raw("VLD1 (R10), [V5.S2]").Raw("SUB $8, R10, R10").Raw("VREV64 V5.S2, V5.S2").Raw("VFCVTL V5.S2, V5.D2")
	} else {
		b.Raw("VLD1.P 16(R6), [V1.D2]")
		b.Raw("VLD1 (R8), [V2.D2]").Raw("SUB $16, R8, R8").Raw("VEXT $8, V2.B16, V2.B16, V2.B16")
		b.Raw("VLD1.P 16(R9), [V4.D2]")
		b.Raw("VLD1 (R10), [V5.D2]").Raw("SUB $16, R10, R10").Raw("VEXT $8, V5.B16, V5.B16, V5.B16")
	}
	b.Raw("VFSUB V2.D2, V1.D2, V3.D2") // C - D
	b.Raw("VFADD V5.D2, V4.D2, V6.D2") // A + B
	b.Raw("VFNEG V6.D2, V6.D2")        // -(B + A)
	b.Raw("VST1.P [V3.D2], 16(R7)")    // dst[h+i]
	b.Raw("VST1.P [V6.D2], 16(R2)")    // dst[i]
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}

// unfold: func(dst, u []float64), dst []float32 when out32.
func unfold(name string, out32 bool) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("u")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R6 S front, R8 T front, R2 dst[i] (T), R9 dst[m+h+i] (-S), R10 dst[m-i-2], R11 dst[m+h-i-2]
	b.LoadArg("dst_base", "R2").LoadArg("u_base", "R0").LoadArg("u_len", "R1")
	ds, step := 3, 16 // destination element shift and bytes per step
	if out32 {
		ds, step = 2, 8
	}
	b.Raw("LSR $2, R1, R14").Raw("CBZ R14, done")
	b.Raw("MOVD R0, R6").Raw("LSL $2, R1, R3").Raw("ADD R3, R0, R8") // T = u + h*8 = u + m*4
	b.Raw(fmt.Sprintf("LSL $%d, R1, R3", ds))                        // m dst bytes
	b.Raw(fmt.Sprintf("LSL $%d, R1, R4", ds-1))                      // h dst bytes
	b.Raw("ADD R3, R4, R5")                                          // (m+h) dst bytes
	b.Raw("ADD R5, R2, R9")
	b.Raw(fmt.Sprintf("SUB $%d, R3, R10", step)).Raw("ADD R2, R10, R10")
	b.Raw(fmt.Sprintf("SUB $%d, R5, R11", step)).Raw("ADD R2, R11, R11")
	b.Label("loop")
	b.Raw("VLD1.P 16(R6), [V1.D2]").Raw("VLD1.P 16(R8), [V2.D2]")
	b.Raw("VFNEG V1.D2, V3.D2").Raw("VFNEG V2.D2, V4.D2")
	b.Raw("VEXT $8, V4.B16, V4.B16, V5.B16").Raw("VEXT $8, V3.B16, V3.B16, V6.B16")
	if out32 {
		b.Raw("VFCVTN V2.D2, V2.S2").Raw("VFCVTN V3.D2, V3.S2").Raw("VFCVTN V5.D2, V5.S2").Raw("VFCVTN V6.D2, V6.S2")
		b.Raw("VST1.P [V2.S2], 8(R2)").Raw("VST1.P [V3.S2], 8(R9)")
		b.Raw("VST1 [V5.S2], (R10)").Raw("SUB $8, R10, R10").Raw("VST1 [V6.S2], (R11)").Raw("SUB $8, R11, R11")
	} else {
		b.Raw("VST1.P [V2.D2], 16(R2)").Raw("VST1.P [V3.D2], 16(R9)")
		b.Raw("VST1 [V5.D2], (R10)").Raw("SUB $16, R10, R10").Raw("VST1 [V6.D2], (R11)").Raw("SUB $16, R11, R11")
	}
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}
