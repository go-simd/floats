//go:build ignore

// Command gen produces round_s390x.s with go-asmgen: the s390x vector-facility
// (z13 baseline) float64 elementwise Abs, Neg, Square, Floor, Ceil, Trunc and
// RoundEven kernels (go-ndarray/ndarray#118).
//
// Sources: z/Architecture Principles of Operation (SA22-7832), Vector
// Floating-Point Instructions, all in the z13 vector facility (not opened when
// this was written; the M5 values are verified by execution under QEMU, and the
// VRR-a byte layout against the bytes cmd/asm emits for VFLPDB):
//
//   - VFPSO (VECTOR FP PERFORM SIGN OPERATION), element size 3 (long), M5 = 0
//     complement, 1 negative, 2 positive: Go names the two it needs VFLCDB
//     (complement: flips the sign, so Neg) and VFLPDB (positive: clears it, so
//     Abs). Sign-bit operations only: a NaN keeps its payload.
//   - VFMDB: one IEEE multiplication per lane, no fusion (Square).
//   - VFI (VECTOR LOAD FP INTEGER, E7C7, VRR-a): M3 = 3 (long), M4 = 0 (all
//     elements, Inexact not suppressed: it only sets a flag, the default FPC masks
//     the trap), M5 the rounding method: 4 round to nearest, ties to even
//     (RoundEven), 5 toward zero (Trunc), 6 toward +Inf (Ceil), 7 toward -Inf
//     (Floor). The result of VFI is exact; a signalling NaN is quieted.
//
// cmd/asm declares VFIDB but has no operand form for it, so it is emitted as the
// six bytes of the VRR-a format, E7 | V1 V2 | 00 | M5 M4 | M3 RXB | C7, as
// WORD + BYTE + BYTE (RXB = 0: only V0..V7 are used).
//
// Operand order, as the other s390x kernels of this repository (measured under
// s390x emulation, the IBM manuals give the other order): `VFMDB V1, V2, V3` is
// V3 = V2 * V1.
//
// Each kernel does n/8 blocks of eight elements (n a positive multiple of 8; the
// Go caller finishes the last seven); VLM loads and VSTM stores four vectors, a
// block is computed before it is stored, so dst may alias src exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	for _, k := range []struct {
		name string
		op   func(r int) string
	}{
		{"absVX", func(r int) string { return fmt.Sprintf("VFLPDB V%d, V%d", r, r) }},
		{"negVX", func(r int) string { return fmt.Sprintf("VFLCDB V%d, V%d", r, r) }},
		{"squareVX", func(r int) string { return fmt.Sprintf("VFMDB V%d, V%d, V%d", r, r, r) }},
		{"floorVX", func(r int) string { return vfi(r, 7) }},
		{"ceilVX", func(r int) string { return vfi(r, 6) }},
		{"truncVX", func(r int) string { return vfi(r, 5) }},
		{"rintVX", func(r int) string { return vfi(r, 4) }},
	} {
		f.Add(kernel(k.name, k.op))
	}
	if err := os.WriteFile("round_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_s390x.s")
}

// vfi is VFIDB V[r], V[r] with M4 = 0 and rounding method m5, as raw bytes.
func vfi(r, m5 int) string {
	return fmt.Sprintf("WORD $0xe7%02x00%x0; BYTE $0x30; BYTE $0xc7 // vfidb v%d, v%d, 0, %d", r<<4|r, m5, r, r, m5)
}

func kernel(name string, op func(r int) string) *emit.Function {
	sig := s390x.Layout([]string{"dst", "src", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R1").LoadArg("src", "R2").LoadArg("n", "R4").
		Raw("loop:").Raw("VLM (R2), V0, V3")
	for r := 0; r < 4; r++ {
		b.Raw("%s", op(r))
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("SUB $8, R4").
		Raw("CMP R4, $0").Raw("BGT loop").Ret()
	return b.Func()
}
