package floats

import "math"

// # Conjugated complex dot product
//
// CDotC is the BLAS zdotc on complex numbers held as consecutive (re, im) float64 pairs:
// with n = len(a)/2 and b holding at least n pairs,
//
//	Σ_{k<n} conj(a_k)·b_k = Σ (ar·br + ai·bi) + i·Σ (ar·bi − ai·br)
//
// It is the lag-k correlation of a complex sequence with itself (b the sequence shifted
// by k, a the sequence): the covariance estimates of a linear predictor. The sum of the
// squared magnitudes of a complex slice is not a new kernel: Dot(a, a) over the same
// float64 view is Σ (re² + im²).
//
// A loop in Go does four multiply-adds per complex number in one dependent chain per
// accumulator; the kernels run two accumulators per lane group (NEON four registers of one
// complex number, AVX2 two registers of two) on a block of four complex numbers, the
// second sum on the real and imaginary parts of b swapped, and fold the lanes at the end.
//
// Contract (go-fft#121, the fleet rule: a stated bound, not bit identity): with u = 2^-53
// and S the sum of the absolute values of the products of a result (|ar br| + |ai bi| for
// the real part, |ar bi| + |ai br| for the imaginary part), each result is within
//
//   - the portable path, one serial chain of 2n fused multiply-adds: (2n+2)·u·S; the bound
//     is reached (an input whose rounding errors all have the same sign gives 2n-2 units),
//     so it cannot be tightened;
//   - the vector paths, chains of about n/4 steps in every lane, then at most four
//     additions to fold the lanes and a serial tail of at most three complex numbers:
//     (n/4+8)·u·S.
//
// The tests assert both against a 300-bit computation for every length from 0 to 70 and at
// the sizes of a transform, on random data, on an adversarial input that reaches the
// portable bound, and on data with massive cancellation. len(a) must be even; b is not read
// past len(a).

// CDotC returns the real and imaginary parts of Σ conj(a_k)·b_k for the n = len(a)/2
// complex numbers a_k = (a[2k], a[2k+1]), b_k = (b[2k], b[2k+1]). It panics if len(a) is
// odd or len(b) < len(a).
func CDotC(a, b []float64) (re, im float64) {
	if len(a)&1 != 0 {
		panic("floats: CDotC: odd len(a)")
	}
	if len(b) < len(a) {
		panic("floats: CDotC: len(b) < len(a)")
	}
	return cdotc(a, b[:len(a)])
}

// cdotcTail is the portable loop from float from on, fused, and the reference of the
// kernels' contract.
func cdotcTail(a, b []float64, from int) (re, im float64) {
	for i := from; i+1 < len(a); i += 2 {
		re = math.FMA(a[i], b[i], math.FMA(a[i+1], b[i+1], re))
		im = math.FMA(a[i], b[i+1], math.FMA(-a[i+1], b[i], im))
	}
	return re, im
}
