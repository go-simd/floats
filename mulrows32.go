package floats

import "math"

// # Row-wise multiply-accumulate in float32
//
// MulAddRows32 is MulAddRows on float32 data: the window stage of a polyphase filter
// bank (an MDCT or QMF synthesis) whose samples are float32, eight (AVX2) or four
// (NEON) to a register instead of four or two,
//
//	dst[j] += Σ_{k<rows} a[k*lda+j] * b[k*ldb+j]        j < len(dst)
//
// with the block of dst held in registers across all rows (AVX2 32, 8 then 1 column,
// NEON 16, 4 then 1), each element of a and b loaded once and dst once.
//
// Contract (a stated error bound, not bit identity with another code): every term is
// a fused multiply-add in increasing k starting from the value dst[j] holds, one
// float32 rounding per step on the kernels. The portable loop computes each term as
// float64 a*b (exact: two 24-bit significands) plus the accumulator, rounded once to
// float32; it can differ from a fused float32 step by one float32 ulp in the rare case
// of a double rounding, so the kernels and the portable loop are compared within the
// bound and not bit for bit. The result is within
//
//	|dst[j] - exact| <= rows * 2^-24 * (|dst0[j]| + Σ_k |a*b|) * (1 + rows * 2^-24)
//
// of the exact value of dst0[j] + Σ a*b (the standard bound of a chain of fused steps;
// asserted against a math/big oracle at every shape by TestMulAddRows32AgainstExact).
//
// Strides lda and ldb are in elements and may be zero; a must hold (rows-1)*lda +
// len(dst) elements and b (rows-1)*ldb + len(dst); nothing past those is read. dst must
// not overlap the part of a or b that is read: checked, panics with a message saying
// so. a and b may overlap each other.

// MulAddRows32 computes dst[j] += Σ_{k<rows} a[k*lda+j]*b[k*ldb+j] for every
// j < len(dst), in float32. rows = 0 or an empty dst do nothing. It panics if rows, lda
// or ldb is negative, if a or b is shorter than the contract says or the index would
// overflow, or if dst overlaps the elements read of a or b.
func MulAddRows32(dst, a []float32, lda int, b []float32, ldb int, rows int) {
	if rows < 0 {
		panic("floats: MulAddRows32: rows < 0")
	}
	if lda < 0 || ldb < 0 {
		panic("floats: MulAddRows32: lda or ldb < 0")
	}
	n := len(dst)
	if rows == 0 || n == 0 {
		return
	}
	na, oka := mulRowsSpan(rows, lda, n)
	nb, okb := mulRowsSpan(rows, ldb, n)
	if !oka || !okb {
		panic("floats: MulAddRows32: rows*stride overflows")
	}
	if len(a) < na || len(b) < nb {
		panic("floats: MulAddRows32: a or b shorter than (rows-1)*stride + len(dst)")
	}
	a, b = a[:na], b[:nb]
	if overlaps(dst, a) || overlaps(dst, b) {
		panic("floats: MulAddRows32: dst overlaps a or b")
	}
	mulAddRows32(dst, a, lda, b, ldb, rows)
}

// mulAddRows32Ref is the portable loop and the reference of the kernels.
func mulAddRows32Ref(dst, a []float32, lda int, b []float32, ldb int, rows int) {
	for j := range dst {
		acc := float64(dst[j])
		for k := 0; k < rows; k++ {
			acc = float64(float32(math.FMA(float64(a[k*lda+j]), float64(b[k*ldb+j]), acc)))
		}
		dst[j] = float32(acc)
	}
}
