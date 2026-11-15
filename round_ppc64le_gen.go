//go:build ignore

// Command gen produces round_ppc64le.s with go-asmgen: the ppc64le VSX (POWER8
// baseline) float64 elementwise Abs, Neg, Square, Floor, Ceil, Trunc and
// RoundEven kernels (go-ndarray/ndarray#118).
//
// Sources: Power ISA Version 3.0B, Book I, VSX Vector Floating-Point Instructions
// (not opened when this was written: the XO values and modes below are from the
// ISA as known and were verified by execution under QEMU POWER9, round_test.go;
// every instruction is in ISA 2.06 and so valid on POWER8), XX2-form: opcode 60, T in bits 6:10 with TX bit 31, B in
// bits 16:20 with BX bit 30, XO in bits 21:29.
//
//   - xvabsdp (XO 473) clears and xvnegdp (XO 505) flips the sign bit of each
//     doubleword, nothing else: a NaN keeps its payload.
//   - xvmuldp: one IEEE multiplication per lane (asmgen's XVMULDP).
//   - xvrdpim (XO 249) rounds toward -Inf (Floor), xvrdpip (XO 233) toward +Inf
//     (Ceil), xvrdpiz (XO 217) toward zero (Trunc), as FRIM, FRIP, FRIZ do for a
//     scalar. xvrdpi (XO 201) is NOT RoundEven: it rounds to nearest with ties
//     AWAY from zero. RoundEven is xvrdpic (XO 235), "round to double-precision
//     integer using current rounding mode": under FPSCR RN = 00 (round to nearest
//     even, the reset value, which Go never changes) it rounds ties to even. All
//     five signal no Inexact and quiet a signalling NaN, and leave zeros and
//     infinities as they are.
//
// Go's ppc64 assembler has none of these mnemonics (it has the loads, stores and
// permutes), so each is a WORD, with its mnemonic as the comment.
//
// Lane order: LXVD2X/STXVD2X swap the doublewords of a vector on little-endian.
// Every vector of one kernel is loaded and stored that way, so a lane-wise
// operation pairs the same elements in every operand and the store puts each
// result back at its own index.
//
// Each kernel does n/8 blocks of eight elements (n a positive multiple of 8; the
// Go caller finishes the last seven); a block is loaded, computed and stored
// before the next, so dst may alias src exactly. R14..R16 hold the byte offsets
// LXVD2X's indexed form needs (R0 reads as 0).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	for _, k := range []struct {
		name, mnem string
		xo         uint32 // 0: xvmuldp
	}{
		{"absVSX", "xvabsdp", 473}, {"negVSX", "xvnegdp", 505}, {"squareVSX", "xvmuldp", 0},
		{"floorVSX", "xvrdpim", 249}, {"ceilVSX", "xvrdpip", 233},
		{"truncVSX", "xvrdpiz", 217}, {"rintVSX", "xvrdpic", 235},
	} {
		f.Add(kernel(k.name, k.mnem, k.xo))
	}
	if err := os.WriteFile("round_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_ppc64le.s")
}

// xx2 encodes an XX2-form instruction: VSR[t] = op(VSR[b]).
func xx2(xo uint32, t, b int) uint32 {
	return 60<<26 | uint32(t&31)<<21 | uint32(b&31)<<11 | xo<<2 | uint32(b>>5)<<1 | uint32(t>>5)
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// kernel builds name(dst, src *float64, n int): VS32..VS35 hold the source,
// VS36..VS39 the result.
func kernel(name, mnem string, xo uint32) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "src", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R3").LoadArg("src", "R5").LoadArg("n", "R4").
		Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16").
		Label("loop")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		if xo == 0 {
			b.XVMULDP(36+k, 32+k, 32+k)
		} else {
			b.Raw("WORD $0x%08x // %s VS%d, VS%d", xx2(xo, 36+k, 32+k), mnem, 36+k, 32+k)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVD2X VS%d, (R3)(%s)", 36+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-8, R4").
		Raw("CMP R4, $0").Raw("BGT loop").Ret()
	return b.Func()
}
