//go:build ignore

// Command gen produces explog_arm64.s with go-asmgen: the NEON kernels of the
// correctly rounded float32 Exp and Log (exp32CRNEON, log32CRNEON: go-ndarray#95,
// proven correctly rounded over all 2^32 float32 inputs), moved from go-ndarray's
// internal/kernels as K1 of the go-scikit performance campaign. The kernels are
// the generator code of go-ndarray#95 unchanged apart from the file they write.
//
// Instructions Go's assembler does not name (FCVTL/FCVTL2, FCVTN/FCVTN2,
// UXTL/UXTL2, SXTL/SXTL2, SCVTF, FACGE, CMGT, BIC) are emitted as WORD
// encodings with the mnemonic in a comment; the tests hold every lane to the Go
// transcription bit for bit, and `go tool objdump -s 'exp32CRNEON|log32CRNEON'`
// on the test binary shows each WORD disassembling to the mnemonic it claims
// (checked after the port to go-simd/floats: VCMGT, VUXTL/VUXTL2, VSXTL/VSXTL2,
// SCVTF, VFCVTN/VFCVTN2, VFCVTL/VFCVTL2, VFACGE, VBIC, all with the operands of
// the comments).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> explog_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(log32CRKernel())
	f.Add(exp32CRKernel())
	if err := os.WriteFile("explog_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote explog_arm64.s")
}

// simd2 emits an Advanced SIMD two-register instruction that Go's assembler
// does not name, by its encoding (Arm ARM, "Advanced SIMD two-register
// miscellaneous"), with the mnemonic as a comment. A wrong encoding cannot
// pass TestExp32NEONMatchesGo, which holds every lane to exp32CR bit for bit.
func simd2(base uint32, mnem string, d, n int) string {
	return fmt.Sprintf("WORD $0x%08X // "+mnem, base|uint32(n)<<5|uint32(d), d, n)
}

// facge emits FACGE Vd.4S, Vn.4S, Vm.4S ("Advanced SIMD three same", U=1,
// opcode 11101): all ones where |Vn| >= |Vm|, zero otherwise and for NaN.
func facge(d, n, m int) string {
	return fmt.Sprintf("WORD $0x%08X // FACGE V%d.4S, V%d.4S, V%d.4S",
		0x6E20EC00|uint32(m)<<16|uint32(n)<<5|uint32(d), d, n, m)
}

const (
	fcvtl  = 0x0E617800 // FCVTL Vd.2D, Vn.2S: the low two floats, widened
	fcvtl2 = 0x4E617800 // FCVTL2 Vd.2D, Vn.4S: the high two
	fcvtn  = 0x0E616800 // FCVTN Vd.2S, Vn.2D: rounded once (FPCR: nearest), upper half zeroed
	fcvtn2 = 0x4E616800 // FCVTN2 Vd.4S, Vn.2D: into the upper half, lower kept
	uxtl   = 0x2F20A400 // UXTL Vd.2D, Vn.2S (USHLL #0): the low two words, zero-extended
	uxtl2  = 0x6F20A400 // UXTL2 Vd.2D, Vn.4S: the high two
	sxtl   = 0x0F20A400 // SXTL Vd.2D, Vn.2S (SSHLL #0): the low two words, sign-extended
	sxtl2  = 0x4F20A400 // SXTL2 Vd.2D, Vn.4S: the high two
	scvtf  = 0x4E61D800 // SCVTF Vd.2D, Vn.2D: int64 to float64 (exact below 2^53)
)

// simd3 emits an Advanced SIMD three-register instruction Go's assembler
// does not name: Vd = Vn OP Vm, with base the encoding's fixed bits.
func simd3(base uint32, mnem string, d, n, m int) string {
	return fmt.Sprintf("WORD $0x%08X // "+mnem, base|uint32(m)<<16|uint32(n)<<5|uint32(d), d, n, m)
}

const (
	cmgt4s = 0x4EA03400 // CMGT Vd.4S, Vn.4S, Vm.4S: Vn > Vm as int32, all ones or zero
	bic16b = 0x4E601C00 // BIC Vd.16B, Vn.16B, Vm.16B: Vn AND NOT Vm
)

// log32CRKernel builds log32CRNEON(dst, src *float32, n int, k
// *logCRNEONConsts) (done int): logCR's fast path (logcr.go, CORE-MATH's
// logf) on eight float32 per iteration, n a positive multiple of 8, the NEON
// twin of log32CRAVX2 (asmgen/amd64/gen.go). Per quad of four floats:
//
//   - in 32-bit lanes, m = ux & 0x7fffff, j = (m + 2^16) >> 17 (at most 64
//     for any ux, so the lookups stay in the 65-entry tables), e = (ux>>23) -
//     127, x == 1, and "not a positive normal" (ux + 0x7f800000 > -0x01000001
//     as int32);
//   - per pair, in float64 with logCR's unfused operations in its order
//     (VFMUL then VFADD, never FMLA): tz = (m | 1023<<23) << 29 (UXTL), e
//     (SXTL, SCVTF), tr[j] and tl[j] through general registers, z, r, and the
//     bounds r and r + 0x1.f06p-33 rounded once (FCVTN, FCVTN2).
//
// A block of 8 with a lane that is not a positive normal, or whose bounds
// round apart (x == 1 excepted: its result is +0), is not stored; the kernel
// returns the count it wrote and the Go wrapper finishes that block with
// logCR. Constants live in V16-V28.
func log32CRKernel() *emit.Function {
	sig := arm64.Layout(
		[]string{"dst", "src", "n", "k"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Ptr},
		[]string{"done"}, []arm64.Type{arm64.Int64},
	)
	const tr, tl = 208, 728 // offsets of the tables in logCRNEONConsts
	// V16 one, V17 ln2, V18 b0, V19 b1, V20 b2, V21 w (float64 pairs); V22
	// mmask, V23 jadd, V24 bias, V25 sub, V26 maxv, V27 oneBits (int32
	// quads); V28 expOr (int64 pair).
	b := arm64.NewFunc("log32CRNEON", sig, 0)
	b.LoadArg("dst", "R0").LoadArg("src", "R1").LoadArg("n", "R2").LoadArg("k", "R8")
	b.Raw("MOVD R8, R13")
	b.Raw("VLD1.P 64(R13), [V16.D2, V17.D2, V18.D2, V19.D2]")
	b.Raw("VLD1.P 64(R13), [V20.D2, V21.D2, V22.D2, V23.D2]")
	b.Raw("VLD1.P 64(R13), [V24.D2, V25.D2, V26.D2, V27.D2]")
	b.Raw("VLD1 (R13), [V28.D2]")
	b.Raw(fmt.Sprintf("ADD $%d, R8, R9", tr))
	b.Raw(fmt.Sprintf("ADD $%d, R8, R10", tl))
	b.Raw("MOVD ZR, R3")
	// quad runs the fast path on the four floats in x, leaving the results
	// (x == 1 lanes as +0) in res and the lanes that may be stored in ok.
	// Integer part in V2-V6, pair temporaries V12-V15 and V29-V31, bounds
	// in V7 (ub) and V8 (lb).
	quad := func(x, res, ok int) {
		b.Raw(fmt.Sprintf("VAND V22.B16, V%d.B16, V2.B16", x)) // m
		b.Raw("VADD V23.S4, V2.S4, V3.S4")
		b.Raw("VUSHR $17, V3.S4, V3.S4") // j
		b.Raw(fmt.Sprintf("VUSHR $23, V%d.S4, V4.S4", x))
		b.Raw("VSUB V24.S4, V4.S4, V4.S4")                   // e
		b.Raw(fmt.Sprintf("VCMEQ V27.S4, V%d.S4, V5.S4", x)) // x == 1
		b.Raw(fmt.Sprintf("VADD V25.S4, V%d.S4, V6.S4", x))
		b.Raw(simd3(cmgt4s, "CMGT V%d.4S, V%d.4S, V%d.4S", 6, 6, 26)) // not a positive normal
		for h := 0; h < 2; h++ {
			var wide, narrow, sx uint32 = uxtl, fcvtn, sxtl
			if h == 1 {
				wide, narrow, sx = uxtl2, fcvtn2, sxtl2
			}
			b.Raw(simd2(wide, "UXTL V%d.2D, V%d.S", 12, 2))
			b.Raw("VORR V28.B16, V12.B16, V12.B16")
			b.Raw("VSHL $29, V12.D2, V12.D2") // tz, in [1,2)
			b.Raw(simd2(sx, "SXTL V%d.2D, V%d.S", 13, 4))
			b.Raw(simd2(scvtf, "SCVTF V%d.2D, V%d.2D", 13, 13)) // e as float64
			b.Raw(fmt.Sprintf("VMOV V3.S[%d], R4", 2*h))
			b.Raw(fmt.Sprintf("VMOV V3.S[%d], R5", 2*h+1))
			b.Raw("MOVD (R9)(R4<<3), R6").Raw("MOVD (R9)(R5<<3), R7")
			b.Raw("MOVD (R10)(R4<<3), R11").Raw("MOVD (R10)(R5<<3), R12")
			b.Raw("VMOV R6, V14.D[0]").Raw("VMOV R7, V14.D[1]")   // tr[j]
			b.Raw("VMOV R11, V15.D[0]").Raw("VMOV R12, V15.D[1]") // tl[j]
			b.Raw("VFMUL V14.D2, V12.D2, V14.D2")                 // tz*tr
			b.Raw("VFSUB V16.D2, V14.D2, V14.D2")                 // z
			b.Raw("VFMUL V14.D2, V14.D2, V29.D2")                 // z2
			b.Raw("VFMUL V17.D2, V13.D2, V31.D2")                 // e*ln2
			b.Raw("VFADD V15.D2, V31.D2, V31.D2")                 // + tl
			b.Raw("VFMUL V18.D2, V14.D2, V12.D2")
			b.Raw("VFADD V12.D2, V31.D2, V31.D2") // + z*b0
			b.Raw("VFMUL V20.D2, V14.D2, V12.D2")
			b.Raw("VFADD V19.D2, V12.D2, V12.D2") // b1 + z*b2
			b.Raw("VFMUL V12.D2, V29.D2, V12.D2")
			b.Raw("VFADD V12.D2, V31.D2, V31.D2") // r
			b.Raw("VFADD V21.D2, V31.D2, V30.D2") // r + 0x1.f06p-33
			b.Raw(simd2(narrow, "FCVTN V%d.S, V%d.2D", 7, 31))
			b.Raw(simd2(narrow, "FCVTN V%d.S, V%d.2D", 8, 30))
		}
		b.Raw(fmt.Sprintf("VCMEQ V8.S4, V7.S4, V%d.S4", ok))        // the bounds agree
		b.Raw(fmt.Sprintf("VORR V5.B16, V%d.B16, V%d.B16", ok, ok)) // or x == 1
		b.Raw(simd3(bic16b, "BIC V%d.16B, V%d.16B, V%d.16B", ok, ok, 6))
		b.Raw(simd3(bic16b, "BIC V%d.16B, V%d.16B, V%d.16B", res, 7, 5)) // x == 1 -> +0
	}
	b.Raw("lloop:")
	b.Raw("VLD1 (R1), [V0.S4, V1.S4]")
	quad(0, 9, 11)
	quad(1, 10, 8)
	b.Raw("VAND V8.B16, V11.B16, V11.B16")
	b.Raw("VMOV V11.D[0], R6")
	b.Raw("VMOV V11.D[1], R7")
	b.Raw("AND R6, R7, R7")
	b.Raw("CMN $1, R7") // every lane may be stored
	b.Raw("BNE lbail")
	b.Raw("VST1 [V9.S4, V10.S4], (R0)")
	b.Raw("ADD $32, R0").Raw("ADD $32, R1").Raw("ADD $8, R3")
	b.Raw("CMP R2, R3").Raw("BLT lloop")
	b.Raw("lbail:")
	b.StoreRet("R3", "done")
	b.Ret()
	return b.Func()
}

// exp32CRKernel builds exp32CRNEON(dst, src *float32, n int, k
// *expCRNEONConsts) (done int): expCR (expcr.go, Arm's correctly rounded
// cr_expf, an AdvSIMD routine in the first place) on eight float32 per
// iteration, n a positive multiple of 8. Each quad is widened to two pairs of
// doubles (FCVTL, FCVTL2); each pair runs expCR's fused operations in its
// order (VFMLA and VFMLS round once, as math.FMA does); the table entries
// z&63 are fetched through two general registers, as cr_expf does, NEON
// having no gather; the result is rounded once (FCVTN, FCVTN2).
//
// A block holding a lane with |x| > 512 or a NaN (FACGE against 512 is false
// for both) is not stored: the kernel returns the count it wrote and the Go
// wrapper finishes that block with exp32CR. Constants live in V16-V26.
func exp32CRKernel() *emit.Function {
	sig := arm64.Layout(
		[]string{"dst", "src", "n", "k"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64, arm64.Ptr},
		[]string{"done"}, []arm64.Type{arm64.Int64},
	)
	const table = 176 // offset of the 64 mantissas in expCRNEONConsts
	b := arm64.NewFunc("exp32CRNEON", sig, 0)
	b.LoadArg("dst", "R0").LoadArg("src", "R1").LoadArg("n", "R2").LoadArg("k", "R8")
	b.Raw("MOVD R8, R10")
	b.Raw("VLD1.P 64(R10), [V16.D2, V17.D2, V18.D2, V19.D2]") // shift, invLn2, ln2Hi, ln2Lo
	b.Raw("VLD1.P 64(R10), [V20.D2, V21.D2, V22.D2, V23.D2]") // c0, c1, c2, c3
	b.Raw("VLD1 (R10), [V24.D2, V25.D2, V26.D2]")             // idxMask, expMask, 512 as 4 floats
	b.Raw(fmt.Sprintf("ADD $%d, R8, R9", table))
	b.Raw("MOVD ZR, R3")
	// half runs expCR on the two doubles in x, leaving the result in t[0].
	// t holds z, n, r2, p01, p23, idx, mant; ra and rb carry the lookup.
	half := func(x int, t [7]int, ra, rb string) {
		z, nn, r2, p01, p23, idx, mant := t[0], t[1], t[2], t[3], t[4], t[5], t[6]
		b.Raw(fmt.Sprintf("VMOV V16.B16, V%d.B16", z))
		b.Raw(fmt.Sprintf("VFMLA V17.D2, V%d.D2, V%d.D2", x, z))     // z = x*invLn2 + shift
		b.Raw(fmt.Sprintf("VFSUB V16.D2, V%d.D2, V%d.D2", z, nn))    // n = z - shift
		b.Raw(fmt.Sprintf("VFMLS V18.D2, V%d.D2, V%d.D2", nn, x))    // r = x - n*ln2Hi
		b.Raw(fmt.Sprintf("VFMLS V19.D2, V%d.D2, V%d.D2", nn, x))    // r -= n*ln2Lo
		b.Raw(fmt.Sprintf("VFMUL V%d.D2, V%d.D2, V%d.D2", x, x, r2)) // r2
		b.Raw(fmt.Sprintf("VMOV V20.B16, V%d.B16", p01))
		b.Raw(fmt.Sprintf("VFMLA V21.D2, V%d.D2, V%d.D2", x, p01)) // p01 = r*c1 + c0
		b.Raw(fmt.Sprintf("VMOV V22.B16, V%d.B16", p23))
		b.Raw(fmt.Sprintf("VFMLA V23.D2, V%d.D2, V%d.D2", x, p23))       // p23 = r*c3 + c2
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", p23, r2, p01)) // p04 = r2*p23 + p01
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", p01, r2, x))   // y = r2*p04 + r
		b.Raw(fmt.Sprintf("VAND V24.B16, V%d.B16, V%d.B16", z, idx))     // z & 63
		b.Raw(fmt.Sprintf("VMOV V%d.D[0], %s", idx, ra))
		b.Raw(fmt.Sprintf("VMOV V%d.D[1], %s", idx, rb))
		b.Raw(fmt.Sprintf("MOVD (R9)(%s<<3), %s", ra, ra))
		b.Raw(fmt.Sprintf("MOVD (R9)(%s<<3), %s", rb, rb))
		b.Raw(fmt.Sprintf("VMOV %s, V%d.D[0]", ra, mant))
		b.Raw(fmt.Sprintf("VMOV %s, V%d.D[1]", rb, mant))
		b.Raw(fmt.Sprintf("VSHL $46, V%d.D2, V%d.D2", z, z))
		b.Raw(fmt.Sprintf("VAND V25.B16, V%d.B16, V%d.B16", z, z))       // exponent and sign
		b.Raw(fmt.Sprintf("VORR V%d.B16, V%d.B16, V%d.B16", mant, z, z)) // s
		b.Raw(fmt.Sprintf("VFMLA V%d.D2, V%d.D2, V%d.D2", x, z, z))      // s*y + s
	}
	t0 := [7]int{3, 4, 5, 6, 7, 14, 15}
	t1 := [7]int{27, 28, 29, 30, 31, 12, 13}
	b.Raw("eloop:")
	b.Raw("VLD1 (R1), [V0.S4, V1.S4]")
	b.Raw(facge(12, 26, 0)) // 512 >= |x|: false for NaN
	b.Raw(facge(13, 26, 1))
	b.Raw("VAND V13.B16, V12.B16, V12.B16")
	b.Raw("VMOV V12.D[0], R6")
	b.Raw("VMOV V12.D[1], R7")
	b.Raw("AND R6, R7, R7")
	b.Raw("CMN $1, R7") // every lane in range
	b.Raw("BNE ebail")
	b.Raw(simd2(fcvtl, "FCVTL V%d.2D, V%d.2S", 2, 0))
	b.Raw(simd2(fcvtl2, "FCVTL2 V%d.2D, V%d.4S", 9, 0))
	b.Raw(simd2(fcvtl, "FCVTL V%d.2D, V%d.2S", 10, 1))
	b.Raw(simd2(fcvtl2, "FCVTL2 V%d.2D, V%d.4S", 11, 1))
	half(2, t0, "R4", "R5")
	half(9, t1, "R6", "R7")
	b.Raw(simd2(fcvtn, "FCVTN V%d.2S, V%d.2D", 0, 3))
	b.Raw(simd2(fcvtn2, "FCVTN2 V%d.4S, V%d.2D", 0, 27))
	half(10, t0, "R4", "R5")
	half(11, t1, "R6", "R7")
	b.Raw(simd2(fcvtn, "FCVTN V%d.2S, V%d.2D", 1, 3))
	b.Raw(simd2(fcvtn2, "FCVTN2 V%d.4S, V%d.2D", 1, 27))
	b.Raw("VST1 [V0.S4, V1.S4], (R0)")
	b.Raw("ADD $32, R0").Raw("ADD $32, R1").Raw("ADD $8, R3")
	b.Raw("CMP R2, R3").Raw("BLT eloop")
	b.Raw("ebail:")
	b.StoreRet("R3", "done")
	b.Ret()
	return b.Func()
}
