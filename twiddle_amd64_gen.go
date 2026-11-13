//go:build ignore

// Command gen produces twiddle_amd64.s with go-asmgen: the AVX2+FMA kernels of the
// DCT-IV twiddle passes (twiddle.go), four complex numbers (two YMM registers of
// float64) per step; the NEON kernels are twiddle_arm64_gen.go, and the blocks
// are the same.
//
// Gather, for the block at float index e = 2m (m .. m+3), n = len(tw):
//
//	x[e : e+8]        -> E  = evens  (x[2m], x[2m+2], x[2m+4], x[2m+6])
//	x[n-8-e : n-e]    -> O' = odds,  O = reverse(O')   (x[n-1-2m] ... x[n-7-2m])
//	tw[e : e+8]       -> Wr, Wi
//	re = E*Wr - O*Wi          im = E*Wi + O*Wr
//	dst[e : e+8]      <- interleave(re, im)
//
// Scatter, for the block of four complex numbers at the front (float index e) and
// the block at the back (float index b = n-8-e, the mirror), z = tw*c for both; the
// front's outputs interleave f*Re zF with reverse(-f*Im zB) and the back's f*Re zB
// with reverse(-f*Im zF); ceil(n/16) steps cover every output (the ends may meet and
// rewrite the same values).
//
// Four doubles in two registers A = (a0 a1 a2 a3), B = (a4 a5 a6 a7) are
// deinterleaved without a constant: VSHUFPD $0 gives (a0 a4 a2 a6), VPERMPD $0xD8
// puts it in order (a0 a2 a4 a6); VSHUFPD $0xF and VPERMPD $0xD8 give the odds, and
// VSHUFPD $0xF then VPERMPD $0x27 gives them reversed (a7 a5 a3 a1). Interleaving
// re and im is VUNPCKLPD / VUNPCKHPD then two VPERM2F128. The float32 input is
// even/odd-selected with VSHUFPS $0x88 / $0xDD and VPERMPD $0x08, then widened with
// VCVTPS2PD; the float32 output is narrowed with VCVTPD2PS and interleaved with
// VUNPCKLPS / VUNPCKHPS. The second product of each sum is fused (VFNMADD231PD /
// VFMADD231PD: one rounding fewer than the portable loop, within the bound stated
// in twiddle.go), so the kernels need FMA: the gate is hasAVX2FMA, as for the other
// AVX2 kernels.
//
// Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`; VSHUFPD $i, Y1, Y0,
// Y2 takes Y0 as its first source.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> twiddle_amd64_gen.go
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
	f.Add(gather("cmulGather64AVX2", false).Func())
	f.Add(gather("cmulGatherFrom32AVX2", true).Func())
	f.Add(scatter("cmulScatter64AVX2", false).Func())
	f.Add(scatter("cmulScatterTo32AVX2", true).Func())
	if err := os.WriteFile("twiddle_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote twiddle_amd64.s")
}

// split4 deinterleaves the 8 doubles in ymm a, b into the evens (ev) and the odds
// (od), each in order; the odds come reversed when rev.
func split4(b *amd64.Builder, a, bb, ev, od string, rev bool) {
	if ev != "" {
		b.Raw("VSHUFPD $0x00, %s, %s, %s", bb, a, ev).Raw("VPERMPD $0xD8, %s, %s", ev, ev)
	}
	if od != "" {
		b.Raw("VSHUFPD $0x0F, %s, %s, %s", bb, a, od)
		if rev {
			b.Raw("VPERMPD $0x27, %s, %s", od, od)
		} else {
			b.Raw("VPERMPD $0xD8, %s, %s", od, od)
		}
	}
}

// interleave4 stores (re0 o0 re1 o1 | re2 o2 re3 o3) for the YMM pair re, od at
// (ptr), four doubles per register, using Y0..Y3 as scratch; ptr is advanced or
// not by the caller.
func interleave4(b *amd64.Builder, re, od, ptr0, ptr1 string) {
	b.Raw("VUNPCKLPD %s, %s, Y0", od, re).Raw("VUNPCKHPD %s, %s, Y1", od, re)
	b.Raw("VPERM2F128 $0x20, Y1, Y0, Y2").Raw("VPERM2F128 $0x31, Y1, Y0, Y3")
	b.Raw("VMOVUPD Y2, %s", ptr0).Raw("VMOVUPD Y3, %s", ptr1)
}

// gather: func(dst, x, tw) with x float64, or func(dst []float64, x []float32, tw
// []float64) when in32.
func gather(name string, in32 bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x"), abi.Slice("tw")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// SI x front, R8 x back, DX tw, DI dst, R10 steps left
	b.LoadArg("dst_base", "DI").LoadArg("x_base", "SI").LoadArg("tw_base", "DX").LoadArg("tw_len", "CX")
	size := 8
	if in32 {
		size = 4
	}
	b.Raw("MOVQ CX, R10").Raw("SHRQ $3, R10").Raw("JZ done")
	b.Raw("LEAQ -8(CX), R11").Raw("LEAQ (SI)(R11*%d), R8", size) // x + (n-8) elements
	b.Label("loop")
	if in32 {
		b.Raw("VMOVUPS (SI), Y0").Raw("ADDQ $32, SI")
		b.Raw("VSHUFPS $0x88, Y0, Y0, Y2").Raw("VPERMPD $0x08, Y2, Y2").Raw("VCVTPS2PD X2, Y2") // E
		b.Raw("VMOVUPS (R8), Y3").Raw("SUBQ $32, R8")
		b.Raw("VSHUFPS $0xDD, Y3, Y3, Y5").Raw("VPERMPD $0x08, Y5, Y5").Raw("VCVTPS2PD X5, Y5").Raw("VPERMPD $0x1B, Y5, Y5") // O reversed
	} else {
		b.Raw("VMOVUPD (SI), Y0").Raw("VMOVUPD 32(SI), Y1").Raw("ADDQ $64, SI")
		split4(b, "Y0", "Y1", "Y2", "", false) // E in Y2
		b.Raw("VMOVUPD (R8), Y3").Raw("VMOVUPD 32(R8), Y4").Raw("SUBQ $64, R8")
		split4(b, "Y3", "Y4", "", "Y5", true) // O reversed in Y5
	}
	b.Raw("VMOVUPD (DX), Y6").Raw("VMOVUPD 32(DX), Y7").Raw("ADDQ $64, DX")
	split4(b, "Y6", "Y7", "Y8", "Y9", false)                    // Wr Y8, Wi Y9
	b.Raw("VMULPD Y8, Y2, Y10").Raw("VFNMADD231PD Y9, Y5, Y10") // re = E*Wr - O*Wi
	b.Raw("VMULPD Y9, Y2, Y11").Raw("VFMADD231PD Y8, Y5, Y11")  // im = E*Wi + O*Wr
	interleave4(b, "Y10", "Y11", "(DI)", "32(DI)")
	b.Raw("ADDQ $64, DI").Raw("DECQ R10").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}

// scatter: func(dst, c, tw []float64, f float64), dst []float32 when out32.
func scatter(name string, out32 bool) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("c"), abi.Slice("tw"), abi.Scalar("f", abi.Float64)}, nil)
	b := amd64.NewFunc(name, sig, 0)
	// SI c front, R8 c back, DX tw front, R9 tw back, DI dst front, R11 dst back,
	// R10 steps left; Y14 = f, Y15 = -f
	b.LoadArg("dst_base", "DI").LoadArg("c_base", "SI").LoadArg("tw_base", "DX").LoadArg("tw_len", "CX").LoadArg("f", "X14")
	b.Raw("VBROADCASTSD X14, Y14").Raw("VXORPD Y15, Y15, Y15").Raw("VSUBPD Y14, Y15, Y15")
	ds := 8
	if out32 {
		ds = 4
	}
	b.Raw("LEAQ 15(CX), R10").Raw("SHRQ $4, R10").Raw("JZ done")
	b.Raw("LEAQ -8(CX), R12")
	b.Raw("LEAQ (SI)(R12*8), R8").Raw("LEAQ (DX)(R12*8), R9").Raw("LEAQ (DI)(R12*%d), R11", ds)
	b.Label("loop")
	// front block: zr in Y8, zi in Y9 (scaled by f and -f)
	b.Raw("VMOVUPD (SI), Y0").Raw("VMOVUPD 32(SI), Y1").Raw("ADDQ $64, SI")
	split4(b, "Y0", "Y1", "Y2", "Y3", false)
	b.Raw("VMOVUPD (DX), Y4").Raw("VMOVUPD 32(DX), Y5").Raw("ADDQ $64, DX")
	split4(b, "Y4", "Y5", "Y6", "Y7", false)
	b.Raw("VMULPD Y2, Y6, Y8").Raw("VFNMADD231PD Y3, Y7, Y8") // zr = Tr*Cr - Ti*Ci
	b.Raw("VMULPD Y3, Y6, Y9").Raw("VFMADD231PD Y2, Y7, Y9")  // zi = Tr*Ci + Ti*Cr
	b.Raw("VMULPD Y14, Y8, Y8").Raw("VMULPD Y15, Y9, Y9")
	// back block: Y12, Y13
	b.Raw("VMOVUPD (R8), Y0").Raw("VMOVUPD 32(R8), Y1").Raw("SUBQ $64, R8")
	split4(b, "Y0", "Y1", "Y2", "Y3", false)
	b.Raw("VMOVUPD (R9), Y4").Raw("VMOVUPD 32(R9), Y5").Raw("SUBQ $64, R9")
	split4(b, "Y4", "Y5", "Y6", "Y7", false)
	b.Raw("VMULPD Y2, Y6, Y12").Raw("VFNMADD231PD Y3, Y7, Y12")
	b.Raw("VMULPD Y3, Y6, Y13").Raw("VFMADD231PD Y2, Y7, Y13")
	b.Raw("VMULPD Y14, Y12, Y12").Raw("VMULPD Y15, Y13, Y13")
	// odd streams: the other block's -f*Im, reversed
	b.Raw("VPERMPD $0x1B, Y13, Y10").Raw("VPERMPD $0x1B, Y9, Y11")
	if out32 {
		b.Raw("VCVTPD2PSY Y8, X8").Raw("VCVTPD2PSY Y10, X10").Raw("VCVTPD2PSY Y12, X12").Raw("VCVTPD2PSY Y11, X11")
		b.Raw("VUNPCKLPS X10, X8, X0").Raw("VUNPCKHPS X10, X8, X1")
		b.Raw("VMOVUPS X0, (DI)").Raw("VMOVUPS X1, 16(DI)").Raw("ADDQ $32, DI")
		b.Raw("VUNPCKLPS X11, X12, X0").Raw("VUNPCKHPS X11, X12, X1")
		b.Raw("VMOVUPS X0, (R11)").Raw("VMOVUPS X1, 16(R11)").Raw("SUBQ $32, R11")
	} else {
		interleave4(b, "Y8", "Y10", "(DI)", "32(DI)")
		b.Raw("ADDQ $64, DI")
		interleave4(b, "Y12", "Y11", "(R11)", "32(R11)")
		b.Raw("SUBQ $64, R11")
	}
	b.Raw("DECQ R10").Raw("JNZ loop")
	b.Label("done").Raw("VZEROUPPER").Ret()
	return b
}
