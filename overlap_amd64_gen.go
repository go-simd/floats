//go:build ignore

// Command gen produces overlap_amd64.s with go-asmgen: the AVX2+FMA kernels of the
// windowed overlap-add (overlap.go), float32 eight lanes and float64 four lanes per
// step.
//
// For n = len(a), block of L lanes at i:
//
//	A  = a[i : i+L]                      Wi = win[i : i+L]
//	Bv = reverse(b[n-i-L : n-i])         Wj = reverse(win[2n-i-L : 2n-i])
//	dst[i : i+L]            = A*Wj - Bv*Wi
//	dst[2n-i-L : 2n-i]      = reverse(A*Wi + Bv*Wj)
//
// A register is reversed without a constant: float64 by VPERMPD $0x1B; float32 by
// VPERMILPS $0x1B (reverse within each 128-bit lane) then VPERM2F128 $0x01 (swap the
// lanes). The second product of each sum is fused (VFNMADD231 / VFMADD231: one
// rounding fewer than the portable loop, within the bound stated in overlap.go),
// so the kernels need FMA: the gate is hasAVX2FMA, as for the other AVX2 kernels.
// The kernel does the blocks that fit, floor(n/L)*L elements; the Go caller
// finishes the tail.
//
// Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`: VMULPS Y4, Y1, Y5
// is Y5 = Y1*Y4; VFMADD231PS Y4, Y3, Y6 is Y6 += Y3*Y4; VFNMADD231PS Y2, Y3, Y5 is
// Y5 -= Y3*Y2.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> overlap_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(kernel("overlapWindow32AVX2", 4, 8, "PS", true).Func())
	f.Add(kernel("overlapWindow64AVX2", 8, 4, "PD", false).Func())
	if err := os.WriteFile("overlap_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote overlap_amd64.s")
}

// kernel: func(dst, a, b, win []T). size is the element size in bytes and the
// index scale, lanes the lanes per YMM register, sfx PS or PD, f32 whether the
// elements are float32.
func kernel(name string, size, lanes int, sfx string, f32 bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b"), abi.Slice("win")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// DI dst, SI a, CX n, DX b, R9 win, AX i, R10 2n
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX").LoadArg("win_base", "R9")
	b.Raw("XORQ AX, AX").Raw("LEAQ (CX)(CX*1), R10")
	reverse := func(y string) {
		if f32 {
			b.Raw("VPERMILPS $0x1B, %s, %s", y, y).Raw("VPERM2F128 $0x01, %s, %s, %s", y, y, y)
		} else {
			b.Raw("VPERMPD $0x1B, %s, %s", y, y)
		}
	}
	b.Label("loop").Raw("LEAQ %d(AX), R8", lanes).Raw("CMPQ R8, CX").Raw("JGT done")
	b.Raw("VMOVUP%s (SI)(AX*%d), Y1", sfx[1:], size)
	b.Raw("VMOVUP%s (R9)(AX*%d), Y2", sfx[1:], size)
	b.Raw("MOVQ CX, R11").Raw("SUBQ AX, R11").Raw("SUBQ $%d, R11", lanes)
	b.Raw("VMOVUP%s (DX)(R11*%d), Y3", sfx[1:], size)
	reverse("Y3")
	b.Raw("MOVQ R10, R12").Raw("SUBQ AX, R12").Raw("SUBQ $%d, R12", lanes)
	b.Raw("VMOVUP%s (R9)(R12*%d), Y4", sfx[1:], size)
	reverse("Y4")
	b.Raw("VMUL%s Y4, Y1, Y5", sfx).Raw("VFNMADD231%s Y2, Y3, Y5", sfx)
	b.Raw("VMUL%s Y2, Y1, Y6", sfx).Raw("VFMADD231%s Y4, Y3, Y6", sfx)
	b.Raw("VMOVUP%s Y5, (DI)(AX*%d)", sfx[1:], size)
	reverse("Y6")
	b.Raw("VMOVUP%s Y6, (DI)(R12*%d)", sfx[1:], size)
	b.Raw("ADDQ $%d, AX", lanes).Raw("JMP loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
