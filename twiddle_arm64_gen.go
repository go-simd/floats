//go:build ignore

// Command gen produces twiddle_arm64.s with go-asmgen: the NEON kernels of the
// DCT-IV twiddle passes (twiddle.go), two complex numbers per step.
//
// Gather, for the block at float index e = 2m (m, m+1), n = len(tw):
//
//	x[e : e+4]        -> E  = evens (x[2m], x[2m+2])
//	x[n-4-e : n-e]    -> O' = odds,  O = reverse(O')   (x[n-1-2m], x[n-3-2m])
//	tw[e : e+4]       -> Wr, Wi
//	re = E*Wr - O*Wi          im = E*Wi + O*Wr
//	dst[e : e+4]      <- interleave(re, im)
//
// Scatter, for the block of two complex numbers at the front (float index e) and
// the block at the back (float index b = n-4-e, the mirror of the front's
// complex indices), z = tw*c for both; the front's outputs are
// [Re zF0, o0, Re zF1, o1] with o = reverse(-f*Im zB) and the back's are
// [Re zB0, p0, Re zB1, p1] with p = reverse(-f*Im zF):
//
//	dst[e : e+4] <- interleave(f*Re zF, reverse(-f*Im zB))
//	dst[b : b+4] <- interleave(f*Re zB, reverse(-f*Im zF))
//
// ceil(n/8) steps cover every output (the two ends may meet and rewrite the same
// values). The reversal of a pair of float64 is VEXT #8.
//
// Deinterleaving is VLD1 of two registers then VUZP1/VUZP2, interleaving is VZIP1/
// VZIP2 then VST1: measured on an Apple M4, 20 percent faster than VLD2/VST2 for the
// same work. The float32 forms read the even or odd pair of one S4 register with
// VUZP1/VUZP2 (VREV64 reverses the odd pair) and widen with VFCVTL; they narrow the
// output with VFCVTN and interleave S4 lanes. The front streams advance with
// post-incremented loads and stores and the back streams by one SUB.
//
// Operand order: Go's arm64 assembler writes three-operand vector forms as
// `OP Vm, Vn, Vd` meaning Vd = Vn op Vm (VUZP1 V2, V1, Vd: the even elements of V1
// then of V2; VZIP1 V2, V1, Vd: V1[0], V2[0], ...); VFMLA / VFMLS accumulate into
// Vd. The second product of each sum is fused (one rounding fewer than the portable
// loop, within the bound stated in twiddle.go).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> twiddle_arm64_gen.go
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
	f.Add(gather("cmulGather64NEON", false).Func())
	f.Add(gather("cmulGatherFrom32NEON", true).Func())
	f.Add(scatter("cmulScatter64NEON", false).Func())
	f.Add(scatter("cmulScatterTo32NEON", true).Func())
	if err := os.WriteFile("twiddle_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote twiddle_arm64.s")
}

// split loads two float64 registers (four values) from ptr, post-incrementing it
// by 32 when post (else leaving it, and the caller steps it), and deinterleaves
// them: even elements to ev, odd elements to od. t0, t1 are consecutive scratch
// registers.
func split(b *arm64.Builder, ptr string, post bool, t0, t1, ev, od int) {
	ld := "VLD1"
	at := fmt.Sprintf("(%s)", ptr)
	if post {
		ld = "VLD1.P"
		at = fmt.Sprintf("32(%s)", ptr)
	}
	b.Raw(fmt.Sprintf("%s %s, [V%d.D2, V%d.D2]", ld, at, t0, t1))
	b.Raw(fmt.Sprintf("VUZP1 V%d.D2, V%d.D2, V%d.D2", t1, t0, ev))
	b.Raw(fmt.Sprintf("VUZP2 V%d.D2, V%d.D2, V%d.D2", t1, t0, od))
}

// gather: func(dst, x, tw) with x float64, or func(dst []float64, x []float32, tw
// []float64) when in32.
func gather(name string, in32 bool) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x"), abi.Slice("tw")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R5 x front, R11 x back, R12 tw, R2 dst, R14 steps left
	b.LoadArg("dst_base", "R2").LoadArg("x_base", "R5").LoadArg("tw_base", "R12").LoadArg("tw_len", "R1")
	xs := 3
	if in32 {
		xs = 2
	}
	b.Raw("SUB $4, R1, R10").Raw(fmt.Sprintf("ADD R10<<%d, R5, R11", xs)) // x + (n-4) elements
	b.Raw("LSR $2, R1, R14").Raw("CBZ R14, done")
	b.Label("loop")
	if in32 {
		// four float32 per side: the even pair from the front, the odd pair from the
		// back reversed, both widened
		b.Raw("VLD1.P 16(R5), [V1.S4]").Raw("VUZP1 V1.S4, V1.S4, V1.S4").Raw("VFCVTL V1.S2, V1.D2")
		b.Raw("VLD1 (R11), [V4.S4]").Raw("SUB $16, R11, R11")
		b.Raw("VUZP2 V4.S4, V4.S4, V4.S4").Raw("VREV64 V4.S2, V4.S2").Raw("VFCVTL V4.S2, V4.D2")
	} else {
		split(b, "R5", true, 1, 2, 16, 2)   // E in V16 (the odds of the front block, V2, are unused)
		split(b, "R11", false, 3, 4, 15, 4) // O' in V4 (V15 unused: the evens of the back block)
		b.Raw("SUB $32, R11, R11")
		b.Raw("VEXT $8, V4.B16, V4.B16, V4.B16") // O (reversed odds) in V4
	}
	split(b, "R12", true, 5, 6, 13, 14) // Wr in V13, Wi in V14
	e := 16                             // E: V16 for float64, V1 widened for float32
	if in32 {
		e = 1
	}
	b.Raw(fmt.Sprintf("VFMUL V13.D2, V%d.D2, V7.D2", e)).Raw("VFMLS V14.D2, V4.D2, V7.D2")
	b.Raw(fmt.Sprintf("VFMUL V14.D2, V%d.D2, V8.D2", e)).Raw("VFMLA V13.D2, V4.D2, V8.D2")
	b.Raw("VZIP1 V8.D2, V7.D2, V9.D2").Raw("VZIP2 V8.D2, V7.D2, V10.D2")
	b.Raw("VST1.P [V9.D2, V10.D2], 32(R2)")
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}

// scatter: func(dst, c, tw []float64, f float64), dst []float32 when out32.
func scatter(name string, out32 bool) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("c"), abi.Slice("tw"), abi.Scalar("f", abi.Float64)}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R5 c front, R11 c back, R6 tw front, R12 tw back, R2 dst front, R4 dst back,
	// R14 steps left
	b.LoadArg("dst_base", "R2").LoadArg("c_base", "R5").LoadArg("tw_base", "R6").LoadArg("tw_len", "R1").LoadArg("f", "F13")
	b.Raw("VDUP V13.D[0], V13.D2").Raw("FNEGD F13, F14").Raw("VDUP V14.D[0], V14.D2")
	ds := 3
	if out32 {
		ds = 2
	}
	b.Raw("SUB $4, R1, R10")
	b.Raw("ADD R10<<3, R5, R11").Raw("ADD R10<<3, R6, R12").Raw(fmt.Sprintf("ADD R10<<%d, R2, R4", ds))
	b.Raw("ADD $7, R1, R14").Raw("LSR $3, R14, R14").Raw("CBZ R14, done")
	b.Label("loop")
	split(b, "R5", true, 1, 2, 19, 20)   // front c: re V19, im V20
	split(b, "R6", true, 3, 4, 21, 22)   // front tw
	split(b, "R11", false, 7, 8, 23, 24) // back c
	b.Raw("SUB $32, R11, R11")
	split(b, "R12", false, 9, 10, 25, 26) // back tw
	b.Raw("SUB $32, R12, R12")
	// zF = tw*c (V5 re, V17 im), zB (V11 re, V18 im)
	b.Raw("VFMUL V21.D2, V19.D2, V5.D2").Raw("VFMLS V22.D2, V20.D2, V5.D2")
	b.Raw("VFMUL V21.D2, V20.D2, V17.D2").Raw("VFMLA V22.D2, V19.D2, V17.D2")
	b.Raw("VFMUL V25.D2, V23.D2, V11.D2").Raw("VFMLS V26.D2, V24.D2, V11.D2")
	b.Raw("VFMUL V25.D2, V24.D2, V18.D2").Raw("VFMLA V26.D2, V23.D2, V18.D2")
	// f*Re, -f*Im
	b.Raw("VFMUL V13.D2, V5.D2, V5.D2").Raw("VFMUL V14.D2, V17.D2, V17.D2")
	b.Raw("VFMUL V13.D2, V11.D2, V11.D2").Raw("VFMUL V14.D2, V18.D2, V18.D2")
	// odd streams: reverse(-f*Im) of the other block
	b.Raw("VEXT $8, V18.B16, V18.B16, V6.B16").Raw("VEXT $8, V17.B16, V17.B16, V12.B16")
	if out32 {
		b.Raw("VFCVTN V5.D2, V5.S2").Raw("VFCVTN V6.D2, V6.S2")
		b.Raw("VFCVTN V11.D2, V11.S2").Raw("VFCVTN V12.D2, V12.S2")
		b.Raw("VZIP1 V6.S4, V5.S4, V27.S4").Raw("VZIP1 V12.S4, V11.S4, V28.S4")
		b.Raw("VST1.P [V27.S4], 16(R2)").Raw("VST1 [V28.S4], (R4)").Raw("SUB $16, R4, R4")
	} else {
		b.Raw("VZIP1 V6.D2, V5.D2, V27.D2").Raw("VZIP2 V6.D2, V5.D2, V28.D2")
		b.Raw("VZIP1 V12.D2, V11.D2, V29.D2").Raw("VZIP2 V12.D2, V11.D2, V30.D2")
		b.Raw("VST1.P [V27.D2, V28.D2], 32(R2)").Raw("VST1 [V29.D2, V30.D2], (R4)").Raw("SUB $32, R4, R4")
	}
	b.Raw("SUBS $1, R14, R14").Raw("BNE loop")
	b.Label("done").Ret()
	return b
}
