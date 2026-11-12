//go:build ignore

// Command gen produces overlap_arm64.s with go-asmgen: the NEON kernels of the
// windowed overlap-add (overlap.go), float32 four lanes and float64 two lanes per
// step.
//
// For n = len(a), block of L lanes at i (j = n-1-i):
//
//	A  = a[i : i+L]                      Wi = win[i : i+L]
//	Bv = reverse(b[n-i-L : n-i])         Wj = reverse(win[2n-i-L : 2n-i])
//	dst[i : i+L]            = A*Wj - Bv*Wi
//	dst[2n-i-L : 2n-i]      = reverse(A*Wi + Bv*Wj)
//
// The reversal of a register is VREV64 (swap the two floats of each 64-bit half,
// float32 only) then VEXT #8 (swap the halves); for float64 VEXT #8 alone. The
// second product of each sum is fused (VFMLS / VFMLA: one rounding fewer than the
// portable loop, within the bound stated in overlap.go). The kernel does the
// blocks that fit, floor(n/L)*L elements; the Go caller finishes the tail.
//
// Operand order: Go's arm64 assembler writes three-operand vector forms as
// `OP Vm, Vn, Vd` meaning Vd = Vn op Vm; VFMLA / VFMLS accumulate into Vd.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> overlap_arm64_gen.go
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
	f.Add(kernel("overlapWindow32NEON", 2, 4, "S4", true).Func())
	f.Add(kernel("overlapWindow64NEON", 3, 2, "D2", false).Func())
	if err := os.WriteFile("overlap_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote overlap_arm64.s")
}

// kernel: func(dst, a, b, win []T). shift is log2 of the element size, lanes the
// lanes per register, arr the arrangement, rev64 whether a register reversal needs
// VREV64 before VEXT.
func kernel(name string, shift, lanes int, arr string, rev64 bool) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b"), abi.Slice("win")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R0 a, R1 n, R2 dst, R7 b, R9 win, R3 i, R13 2n
	b.LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R7").LoadArg("win_base", "R9")
	b.Raw("MOVD $0, R3").Raw("LSL $1, R1, R13")
	reverse := func(v int) {
		if rev64 {
			b.Raw(fmt.Sprintf("VREV64 V%d.%s, V%d.%s", v, arr, v, arr))
		}
		b.Raw(fmt.Sprintf("VEXT $8, V%d.B16, V%d.B16, V%d.B16", v, v, v))
	}
	b.Label("loop").
		Raw(fmt.Sprintf("ADD $%d, R3, R4", lanes)).Raw("CMP R1, R4").Raw("BGT done")
	// A, Wi from the front
	b.Raw(fmt.Sprintf("ADD R3<<%d, R0, R5", shift)).Raw(fmt.Sprintf("VLD1 (R5), [V1.%s]", arr))
	b.Raw(fmt.Sprintf("ADD R3<<%d, R9, R5", shift)).Raw(fmt.Sprintf("VLD1 (R5), [V2.%s]", arr))
	// Bv from n-i-L of b, reversed
	b.Raw("SUB R3, R1, R10").Raw(fmt.Sprintf("SUB $%d, R10, R10", lanes))
	b.Raw(fmt.Sprintf("ADD R10<<%d, R7, R11", shift)).Raw(fmt.Sprintf("VLD1 (R11), [V3.%s]", arr))
	reverse(3)
	// Wj from 2n-i-L of win, reversed
	b.Raw("SUB R3, R13, R12").Raw(fmt.Sprintf("SUB $%d, R12, R12", lanes))
	b.Raw(fmt.Sprintf("ADD R12<<%d, R9, R14", shift)).Raw(fmt.Sprintf("VLD1 (R14), [V4.%s]", arr))
	reverse(4)
	// D1 = A*Wj - Bv*Wi (V5), D2 = A*Wi + Bv*Wj (V6)
	b.Raw(fmt.Sprintf("VFMUL V4.%s, V1.%s, V5.%s", arr, arr, arr))
	b.Raw(fmt.Sprintf("VFMLS V2.%s, V3.%s, V5.%s", arr, arr, arr))
	b.Raw(fmt.Sprintf("VFMUL V2.%s, V1.%s, V6.%s", arr, arr, arr))
	b.Raw(fmt.Sprintf("VFMLA V4.%s, V3.%s, V6.%s", arr, arr, arr))
	// D1 to the front, reversed D2 to the back
	b.Raw(fmt.Sprintf("ADD R3<<%d, R2, R5", shift)).Raw(fmt.Sprintf("VST1 [V5.%s], (R5)", arr))
	reverse(6)
	b.Raw(fmt.Sprintf("ADD R12<<%d, R2, R15", shift)).Raw(fmt.Sprintf("VST1 [V6.%s], (R15)", arr))
	b.Raw(fmt.Sprintf("ADD $%d, R3, R3", lanes)).Raw("B loop")
	b.Label("done").Ret()
	return b
}
