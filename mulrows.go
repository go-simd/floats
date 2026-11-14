package floats

import "math"

// # Row-wise multiply-accumulate (a polyphase window sum)
//
// MulAddRows is the inner loop of a polyphase filter bank (the window stage of an MDCT
// or QMF synthesis): output j is a sum of products of two matrices read down a column,
//
//	dst[j] += Σ_{k<rows} a[k*lda+j] * b[k*ldb+j]        j < len(dst)
//
// A loop in Go over the rows loads and stores the accumulator row by row; the kernel
// holds a block of dst in registers across all rows (NEON 16, 4, 2 then 1 column, AVX2
// 16, 4 then 1), so each element of a and b is loaded once and dst once.
//
// Contract: every element is a chain of fused multiply-adds in increasing k, starting
// from the value dst[j] holds: dst[j] = math.FMA(a[k*lda+j], b[k*ldb+j], dst[j]) for
// k = 0, 1, ... — one rounding per step, bit-identical on every architecture and
// whatever CPU features are found at run time (the kernel and the portable math.FMA
// loop do the same operations in the same order, as Axpy does). It is not the unfused
// dst[j] + a*b of a Go loop on amd64: a caller replacing such a loop may see last-bit
// differences, the fused form being the more accurate one.
//
// Strides lda and ldb are in elements and may be zero (every row is the same row); a
// must hold (rows-1)*lda + len(dst) elements and b (rows-1)*ldb + len(dst); nothing
// past those is read. dst must not overlap the part of a or b that is read, not even
// exactly: checked, panics with a message saying so. a and b may overlap each other.

// MulAddRows computes dst[j] += Σ_{k<rows} a[k*lda+j]*b[k*ldb+j] for every j < len(dst),
// each term a fused multiply-add in increasing k. rows = 0 or an empty dst do nothing.
// It panics if rows, lda or ldb is negative, if a or b is shorter than the contract
// says or the index would overflow, or if dst overlaps the elements read of a or b.
func MulAddRows(dst, a []float64, lda int, b []float64, ldb int, rows int) {
	if rows < 0 {
		panic("floats: MulAddRows: rows < 0")
	}
	if lda < 0 || ldb < 0 {
		panic("floats: MulAddRows: lda or ldb < 0")
	}
	n := len(dst)
	if rows == 0 || n == 0 {
		return
	}
	na, oka := mulRowsSpan(rows, lda, n)
	nb, okb := mulRowsSpan(rows, ldb, n)
	if !oka || !okb {
		panic("floats: MulAddRows: rows*stride overflows")
	}
	if len(a) < na || len(b) < nb {
		panic("floats: MulAddRows: a or b shorter than (rows-1)*stride + len(dst)")
	}
	a, b = a[:na], b[:nb]
	if overlaps(dst, a) || overlaps(dst, b) {
		panic("floats: MulAddRows: dst overlaps a or b")
	}
	mulAddRows(dst, a, lda, b, ldb, rows)
}

// mulRowsSpan is (rows-1)*ld + n, the elements a matrix of rows rows read, and whether
// it fits an int.
func mulRowsSpan(rows, ld, n int) (int, bool) {
	if ld != 0 && rows-1 > (math.MaxInt-n)/ld {
		return 0, false
	}
	return (rows-1)*ld + n, true
}

// mulAddRowsRef is the portable loop and the reference of the kernels.
func mulAddRowsRef(dst, a []float64, lda int, b []float64, ldb int, rows int) {
	for j := range dst {
		acc := dst[j]
		for k := 0; k < rows; k++ {
			acc = math.FMA(a[k*lda+j], b[k*ldb+j], acc)
		}
		dst[j] = acc
	}
}
