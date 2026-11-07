//go:build ignore

// Command gen produces explog_amd64.s with go-asmgen: the AVX2 kernels of the
// correctly rounded float32 Exp and Log (exp32CRAVX2, log32CRAVX2: go-ndarray#94)
// and of the float64 Log (logAVX2: go-ndarray#96), moved from go-ndarray's
// internal/kernels as K1 of the go-scikit performance campaign. The kernels are
// the generator code of go-ndarray unchanged apart from the file they write.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> explog_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(exp32CRKernel())
	f.Add(log32CRKernel())
	f.Add(logKernel())
	if err := os.WriteFile("explog_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote explog_amd64.s")
}

// exp32CRKernel builds exp32CRAVX2(dst, src *float32, n int, k *expCRConsts)
// for n a positive multiple of 8: dst[i] = float32(expCR(clamp(src[i]))), the
// correctly rounded float32 exp, bit for bit the Go expCR (kernels/expcr.go,
// a transcription of Arm optimized-routines' cr_expf).
//
// Eight floats per iteration: clamped to [-512, 512] (exp of the clamped
// value rounds to the same float32 +Inf or 0 as the input's), widened to two
// groups of four doubles, and each group run through expCR's operations in
// the same order and with the same fusing: z = x*invln2 + shift, n = z -
// shift, r = x - n*ln2hi - n*ln2lo (VFNMADD231PD), the degree-5 polynomial,
// the scale assembled from the table entry z&63 (VPGATHERQQ) and the
// exponent bits z<<46, and s*y + s; then rounded once to float32. NaN lanes
// are restored from the input at the end (the clamp turned them into -512).
//
// The constants are 4-lane vectors in the Go block k (offsets below), used as
// memory operands, so all sixteen YMM registers stay free. Needs AVX2 (the
// gather) and FMA.
func exp32CRKernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "src", "n", "k"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Ptr},
		nil, nil,
	)
	const (
		shift, invLn2, ln2Hi, ln2Lo = 0, 32, 64, 96
		c0, c1, c2, c3              = 128, 160, 192, 224
		idxMask, expMask            = 256, 288
		lo, hi                      = 320, 352
		table                       = 384
	)
	b := amd64.NewFunc("exp32CRAVX2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("src", "SI").LoadArg("n", "CX").LoadArg("k", "R8")
	b.Raw(fmt.Sprintf("LEAQ %d(R8), R9", table)) // the 64 mantissas of 2^(i/64)
	// half runs expCR on the four doubles in h, leaving the four float32
	// results in out. Temporaries Y4-Y10.
	half := func(h, out string) {
		b.Raw(fmt.Sprintf("VMOVUPD %d(R8), Y4", shift))
		b.Raw(fmt.Sprintf("VFMADD231PD %d(R8), %s, Y4", invLn2, h)) // z = x*invln2 + shift
		b.Raw(fmt.Sprintf("VSUBPD %d(R8), Y4, Y5", shift))          // n = z - shift
		b.Raw(fmt.Sprintf("VFNMADD231PD %d(R8), Y5, %s", ln2Hi, h)) // r = x - n*ln2hi
		b.Raw(fmt.Sprintf("VFNMADD231PD %d(R8), Y5, %s", ln2Lo, h)) // r -= n*ln2lo
		b.Raw(fmt.Sprintf("VMULPD %s, %s, Y6", h, h))               // r2
		b.Raw(fmt.Sprintf("VMOVUPD %d(R8), Y7", c0))
		b.Raw(fmt.Sprintf("VFMADD231PD %d(R8), %s, Y7", c1, h)) // p01 = r*c1 + c0
		b.Raw(fmt.Sprintf("VMOVUPD %d(R8), Y8", c2))
		b.Raw(fmt.Sprintf("VFMADD231PD %d(R8), %s, Y8", c3, h)) // p23 = r*c3 + c2
		b.Raw("VFMADD231PD Y8, Y6, Y7")                         // p04 = r2*p23 + p01
		b.Raw(fmt.Sprintf("VFMADD231PD Y7, Y6, %s", h))         // y = r2*p04 + r
		b.Raw(fmt.Sprintf("VPAND %d(R8), Y4, Y5", idxMask))     // z & 63
		b.Raw("VPCMPEQQ Y9, Y9, Y9")                            // gather every lane
		b.Raw("VPGATHERQQ Y9, (R9)(Y5*8), Y10")                 // mantissa of 2^(i/64)
		b.Raw("VPSLLQ $46, Y4, Y4")
		b.Raw(fmt.Sprintf("VPAND %d(R8), Y4, Y4", expMask)) // exponent and sign bits
		b.Raw("VPOR Y10, Y4, Y4")                           // s
		b.Raw(fmt.Sprintf("VFMADD231PD %s, Y4, Y4", h))     // s*y + s
		b.Raw(fmt.Sprintf("VCVTPD2PSY Y4, %s", out))        // round once to float32
	}
	b.Raw("eloop:")
	b.Raw("VMOVUPS (SI), Y0")
	b.Raw("VCMPPS $3, Y0, Y0, Y15") // unordered: the NaN lanes
	b.Raw(fmt.Sprintf("VMAXPS %d(R8), Y0, Y1", lo))
	b.Raw(fmt.Sprintf("VMINPS %d(R8), Y1, Y1", hi))
	b.Raw("VCVTPS2PD X1, Y2")
	b.Raw("VEXTRACTF128 $1, Y1, X3")
	b.Raw("VCVTPS2PD X3, Y3")
	half("Y2", "X11")
	half("Y3", "X12")
	b.Raw("VINSERTF128 $1, X12, Y11, Y11")
	b.Raw("VBLENDVPS Y15, Y0, Y11, Y11") // NaN lanes: the input NaN
	b.Raw("VMOVUPS Y11, (DI)")
	b.Raw("ADDQ $32, SI").Raw("ADDQ $32, DI").Raw("SUBQ $8, CX").Raw("JNZ eloop")
	b.Raw("VZEROUPPER")
	b.Ret()
	return b.Func()
}

// log32CRKernel builds log32CRAVX2(dst, src *float32, n int, k
// *logCRConsts) int: logCR's fast path (logcr.go, CORE-MATH's logf) on eight
// float32 at a time, n a positive multiple of 8. Per block:
//
//   - the integer part in 32-bit lanes: m = ux & 0x7fffff, j = (m+2^16)>>17
//     (at most 64 for any ux, so the gathers stay in the 65-entry tables),
//     e = (ux>>23) - 127;
//   - per four lanes, in float64 with logCR's unfused operations in its order
//     (VMULPD then VADDPD, never FMA): tz from (m | 1023<<23)<<29, tr[j] and
//     tl[j] gathered (VGATHERDPD), z = tz*tr - 1, r, and the bounds r and
//     r + 0x1.f06p-33 rounded to float32;
//   - x == 1 is 0 (logCR's early return).
//
// A block with an input that is not a positive normal float32, or a lane
// whose bounds round apart, is not stored: the kernel returns the count of
// floats it wrote, and the Go wrapper finishes that block with logCR. Needs
// AVX2 (gathers, 256-bit integer ops).
func log32CRKernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "src", "n", "k"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Ptr},
		[]string{"done"}, []amd64.Type{amd64.Int64},
	)
	const (
		one, ln2, b0, b1, b2, w               = 0, 32, 64, 96, 128, 160
		sub, maxv, mmask, jadd, bias, oneBits = 192, 224, 256, 288, 320, 352
		expOr                                 = 384
		tr, tl                                = 416, 936
	)
	b := amd64.NewFunc("log32CRAVX2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("src", "SI").LoadArg("n", "CX").LoadArg("k", "R8")
	b.Raw(fmt.Sprintf("LEAQ %d(R8), R9", tr))
	b.Raw(fmt.Sprintf("LEAQ %d(R8), R10", tl))
	b.Raw("XORQ AX, AX")
	// half runs the float64 part on the four lanes whose j is in X2, mantissa
	// bits (zero-extended to 64) in Y10 and exponent (as float64) in Y11,
	// leaving the bounds in ub and lb. Temporaries Y4-Y9, Y12.
	half := func(ub, lb string) {
		b.Raw(fmt.Sprintf("VPOR %d(R8), Y10, Y10", expOr))
		b.Raw("VPSLLQ $29, Y10, Y10")           // tz, in [1,2)
		b.Raw("VPCMPEQQ Y12, Y12, Y12")         // gather every lane
		b.Raw("VGATHERDPD Y12, (R9)(X2*8), Y4") // tr[j]
		b.Raw("VPCMPEQQ Y12, Y12, Y12")
		b.Raw("VGATHERDPD Y12, (R10)(X2*8), Y5")          // tl[j]
		b.Raw("VMULPD Y4, Y10, Y4")                       // tz*tr
		b.Raw(fmt.Sprintf("VSUBPD %d(R8), Y4, Y4", one))  // z
		b.Raw("VMULPD Y4, Y4, Y6")                        // z2
		b.Raw(fmt.Sprintf("VMULPD %d(R8), Y11, Y7", ln2)) // e*ln2
		b.Raw("VADDPD Y5, Y7, Y7")                        // + tl
		b.Raw(fmt.Sprintf("VMULPD %d(R8), Y4, Y8", b0))
		b.Raw("VADDPD Y8, Y7, Y7") // + z*b0
		b.Raw(fmt.Sprintf("VMULPD %d(R8), Y4, Y8", b2))
		b.Raw(fmt.Sprintf("VADDPD %d(R8), Y8, Y8", b1)) // b1 + z*b2
		b.Raw("VMULPD Y8, Y6, Y8")
		b.Raw("VADDPD Y8, Y7, Y7")                     // r
		b.Raw(fmt.Sprintf("VADDPD %d(R8), Y7, Y9", w)) // r + 0x1.f06p-33
		b.Raw(fmt.Sprintf("VCVTPD2PSY Y7, %s", ub))
		b.Raw(fmt.Sprintf("VCVTPD2PSY Y9, %s", lb))
	}
	b.Raw("lloop:")
	b.Raw("VMOVDQU (SI), Y0")
	// Lanes that are not positive normals run through the arithmetic below
	// on meaningless values (j stays at most 64, so the gathers stay inside
	// the tables) and are rejected at the end of the block.
	b.Raw(fmt.Sprintf("VPCMPEQD %d(R8), Y0, Y13", oneBits)) // x == 1
	b.Raw(fmt.Sprintf("VPAND %d(R8), Y0, Y1", mmask))       // m
	b.Raw(fmt.Sprintf("VPADDD %d(R8), Y1, Y2", jadd))
	b.Raw("VPSRLD $17, Y2, Y2") // j
	b.Raw("VPSRLD $23, Y0, Y3")
	b.Raw(fmt.Sprintf("VPSUBD %d(R8), Y3, Y3", bias)) // e
	b.Raw("VPMOVZXDQ X1, Y10")
	b.Raw("VCVTDQ2PD X3, Y11")
	half("X14", "X15")
	b.Raw("VEXTRACTI128 $1, Y1, X1")
	b.Raw("VPMOVZXDQ X1, Y10")
	b.Raw("VEXTRACTI128 $1, Y3, X3")
	b.Raw("VCVTDQ2PD X3, Y11")
	b.Raw("VEXTRACTI128 $1, Y2, X2")
	half("X1", "X3")
	b.Raw("VINSERTI128 $1, X1, Y14, Y14") // ub, eight lanes
	b.Raw("VINSERTI128 $1, X3, Y15, Y15") // lb
	b.Raw("VPCMPEQD Y15, Y14, Y15")       // the bounds agree
	b.Raw("VPOR Y13, Y15, Y15")           // or x == 1
	// Positive normals, ux in [2^23, 0x7f7fffff], are exactly the lanes with
	// ux + 0x7f800000 (wrapping) at most -0x01000001 as int32.
	b.Raw(fmt.Sprintf("VPADDD %d(R8), Y0, Y4", sub))
	b.Raw(fmt.Sprintf("VPCMPGTD %d(R8), Y4, Y4", maxv)) // not a positive normal
	b.Raw("VPANDN Y15, Y4, Y15")
	b.Raw("VMOVMSKPS Y15, DX")
	b.Raw("CMPL DX, $0xff")
	b.Raw("JNE lbail")
	b.Raw("VPANDN Y14, Y13, Y14") // x == 1 -> +0
	b.Raw("VMOVDQU Y14, (DI)")
	b.Raw("ADDQ $32, SI").Raw("ADDQ $32, DI").Raw("ADDQ $8, AX")
	b.Raw("CMPQ AX, CX").Raw("JLT lloop")
	b.Raw("lbail:")
	b.Raw("VZEROUPPER")
	b.StoreRet("AX", "done")
	b.Ret()
	return b.Func()
}

// logKernel builds logAVX2(dst, src *float64, n int, k *logConsts, tab
// *logEntry) (done int): the table path of Log (log.go, the port of
// optimized-routines' log) on four doubles at a time, n a positive multiple
// of 4, with Log's operations in its order and its single FMA
// (r = z*invc - 1, VFMADD231PD) where it has one; every other product and
// sum is a separate VMULPD/VADDPD, as Go compiles them on amd64.
//
//   - tmp = ix - off; i = (tmp>>45) & 127 indexes the 16-byte table entries
//     (VPGATHERQQ at 2i, scale 8: invc, then logc 8 bytes on); k =
//     int64(tmp)>>52 has no AVX2 instruction (no 64-bit arithmetic shift, no
//     int64 to float64 conversion), so kd is built exactly from u = tmp>>52:
//     float64 bits 0x4330000000000000 | (u ^ 0x800) is 2^52 + (u ^ 0x800),
//     and subtracting 2^52 + 2048 leaves the sign-extended k.
//
// A block holding a lane that is not a positive normal (ix < 0x0010<<48 or
// above the largest finite, as int64), or one within 2^-4 of 1 (Log's
// polynomial path), is not stored: the kernel returns the count it wrote and
// Log finishes that block in Go. Needs AVX2 (gathers, 64-bit compares) and
// FMA.
func logKernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "src", "n", "k", "tab"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64, amd64.Ptr, amd64.Ptr},
		[]string{"done"}, []amd64.Type{amd64.Int64},
	)
	const (
		lo, hiMax, n1loM1, n1hi = 0, 32, 64, 96
		off, idxMask, topMask   = 128, 160, 192
		magic, magicSub, x2048  = 224, 256, 288
		negOne, ln2hi, ln2lo    = 320, 352, 384
		a0, a1, a2, a3, a4      = 416, 448, 480, 512, 544
	)
	b := amd64.NewFunc("logAVX2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("src", "SI").LoadArg("n", "CX").LoadArg("k", "R8").LoadArg("tab", "R9")
	b.Raw("XORQ AX, AX")
	b.Raw("dloop:")
	b.Raw("VMOVDQU (SI), Y0") // ix
	// The lanes Log's loop sends to log1.
	b.Raw(fmt.Sprintf("VPCMPGTQ %d(R8), Y0, Y1", hiMax)) // ix > largest finite (as int64)
	b.Raw(fmt.Sprintf("VMOVDQU %d(R8), Y2", lo))
	b.Raw("VPCMPGTQ Y0, Y2, Y2") // ix < 0x0010<<48: subnormal, zero or negative
	b.Raw(fmt.Sprintf("VPCMPGTQ %d(R8), Y0, Y3", n1loM1))
	b.Raw(fmt.Sprintf("VMOVDQU %d(R8), Y4", n1hi))
	b.Raw("VPCMPGTQ Y0, Y4, Y4")
	b.Raw("VPAND Y4, Y3, Y3") // within 2^-4 of 1
	b.Raw("VPOR Y3, Y2, Y2")
	b.Raw("VPOR Y1, Y2, Y2")
	b.Raw("VPTEST Y2, Y2")
	b.Raw("JNZ dbail")
	b.Raw(fmt.Sprintf("VPSUBQ %d(R8), Y0, Y1", off)) // tmp
	b.Raw("VPSRLQ $45, Y1, Y2")
	b.Raw(fmt.Sprintf("VPAND %d(R8), Y2, Y2", idxMask))
	b.Raw("VPADDQ Y2, Y2, Y2") // 2i: entries are two float64 wide
	b.Raw("VPSRLQ $52, Y1, Y3")
	b.Raw(fmt.Sprintf("VPXOR %d(R8), Y3, Y3", x2048))
	b.Raw(fmt.Sprintf("VPOR %d(R8), Y3, Y3", magic))
	b.Raw(fmt.Sprintf("VSUBPD %d(R8), Y3, Y3", magicSub)) // kd
	b.Raw(fmt.Sprintf("VPAND %d(R8), Y1, Y4", topMask))
	b.Raw("VPSUBQ Y4, Y0, Y4") // iz = ix - tmp&(0xfff<<52)
	b.Raw("VPCMPEQQ Y5, Y5, Y5")
	b.Raw("VPGATHERQQ Y5, (R9)(Y2*8), Y6") // invc
	b.Raw("VPCMPEQQ Y5, Y5, Y5")
	b.Raw("VPGATHERQQ Y5, 8(R9)(Y2*8), Y7") // logc
	b.Raw(fmt.Sprintf("VMOVUPD %d(R8), Y8", negOne))
	b.Raw("VFMADD231PD Y6, Y4, Y8")                    // r = z*invc - 1, one rounding
	b.Raw(fmt.Sprintf("VMULPD %d(R8), Y3, Y9", ln2hi)) // kd*ln2hi
	b.Raw("VADDPD Y7, Y9, Y9")                         // w = ... + logc
	b.Raw("VADDPD Y8, Y9, Y10")                        // hi = w + r
	b.Raw("VSUBPD Y10, Y9, Y11")                       // w - hi
	b.Raw("VADDPD Y8, Y11, Y11")                       // + r
	b.Raw(fmt.Sprintf("VMULPD %d(R8), Y3, Y12", ln2lo))
	b.Raw("VADDPD Y12, Y11, Y11") // lo = w - hi + r + kd*ln2lo
	b.Raw("VMULPD Y8, Y8, Y12")   // r2
	b.Raw(fmt.Sprintf("VMULPD %d(R8), Y8, Y13", a2))
	b.Raw(fmt.Sprintf("VADDPD %d(R8), Y13, Y13", a1)) // A1 + r*A2
	b.Raw(fmt.Sprintf("VMULPD %d(R8), Y8, Y14", a4))
	b.Raw(fmt.Sprintf("VADDPD %d(R8), Y14, Y14", a3)) // A3 + r*A4
	b.Raw("VMULPD Y14, Y12, Y14")
	b.Raw("VADDPD Y14, Y13, Y13") // A1 + r*A2 + r2*(A3 + r*A4)
	b.Raw("VMULPD Y12, Y8, Y14")  // r*r2
	b.Raw("VMULPD Y13, Y14, Y14") // r*r2*(...)
	b.Raw(fmt.Sprintf("VMULPD %d(R8), Y12, Y13", a0))
	b.Raw("VADDPD Y13, Y11, Y11") // lo + r2*A0
	b.Raw("VADDPD Y14, Y11, Y11") // + r*r2*(...)
	b.Raw("VADDPD Y10, Y11, Y11") // + hi
	b.Raw("VMOVUPD Y11, (DI)")
	b.Raw("ADDQ $32, SI").Raw("ADDQ $32, DI").Raw("ADDQ $4, AX")
	b.Raw("CMPQ AX, CX").Raw("JLT dloop")
	b.Raw("dbail:")
	b.Raw("VZEROUPPER")
	b.StoreRet("AX", "done")
	b.Ret()
	return b.Func()
}
