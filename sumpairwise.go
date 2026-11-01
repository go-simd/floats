package floats

import "math"

// pwBlock is NumPy's PW_BLOCKSIZE: a pairwise-summation block of at most this
// many elements is summed with eight independent accumulators; a longer one is
// split in two. (numpy/_core/src/umath/loops_utils.h.src, v2.5.3.)
const pwBlock = 128

// SumPairwise returns Σ aᵢ computed exactly as NumPy computes np.sum(a) for a
// contiguous one-dimensional float64 array: the same pairwise summation, with
// the same block size, the same eight-accumulator unrolling and the same order
// of partial sums. The result is bit-identical to np.sum(a) — on every
// architecture, because the order is fixed by this code rather than by the
// width of a vector register, and no step is a fused multiply-add.
//
// It is the function to use when a result must be reproducible against NumPy
// (for instance a stopping test that compares np.std of a population with a
// tolerance). It is not the same as Sum, whose lane-blocked order trades
// reproducibility for speed on the SIMD kernels.
//
// The sum of an empty slice is +0, and so is the sum of a slice holding only
// -0 (the reduction starts from +0), as in NumPy. NaN and ±Inf propagate by IEEE 754 rules; when several NaNs are
// summed which payload survives is, as in NumPy, up to the hardware.
//
// The implementation is scalar Go. NumPy's partial sums r[0..7] are eight
// separate additions per step, which a four-lane vector would reproduce, but a
// SIMD version has to keep that exact association and the gain over the
// unrolled scalar loop below did not justify one assembly kernel per
// architecture; see the README for the measured ratio against np.sum.
func SumPairwise(a []float64) float64 {
	// np.add.reduce starts the accumulator at the identity +0 and adds the
	// pairwise sum of all n elements to it (DOUBLE_add:
	// *dst += pairwise_sum(src1, len), loops_arithm_fp.dispatch.c.src). So the
	// sum of only -0 values is +0, whatever n. (Checked against NumPy 2.5.4:
	// copying a[0] as the initial value and summing a[1:] instead is off by an
	// ulp already at n = 24.)
	return pwInit + pairwise(a)
}

// pwInit is the +0 identity of the reduction. It is a variable so that the
// compiler cannot fold pwInit + x to x, which would be wrong for x = -0.
var pwInit = 0.0

// pwNegZero is -0.0. The literal -0.0 in Go source is +0, so it is built at run
// time; NumPy starts a short block from -0 so that a block of -0 sums to -0.
var pwNegZero = math.Copysign(0, -1)

// pairwise is DOUBLE_pairwise_sum for a contiguous slice.
func pairwise(a []float64) float64 {
	n := len(a)
	if n < 8 {
		res := pwNegZero
		for _, v := range a {
			res += v
		}
		return res
	}
	if n <= pwBlock {
		r0, r1, r2, r3 := a[0], a[1], a[2], a[3]
		r4, r5, r6, r7 := a[4], a[5], a[6], a[7]
		i := 8
		for ; i < n-n%8; i += 8 {
			b := a[i : i+8 : i+8]
			r0 += b[0]
			r1 += b[1]
			r2 += b[2]
			r3 += b[3]
			r4 += b[4]
			r5 += b[5]
			r6 += b[6]
			r7 += b[7]
		}
		res := ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	}
	// Divide by two but avoid non-multiples of the unroll factor.
	n2 := n / 2
	n2 -= n2 % 8
	return pairwise(a[:n2]) + pairwise(a[n2:])
}
