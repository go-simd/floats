package floats

// # Squared magnitude of complex numbers (the power pass of a periodogram)
//
// The last pass of a power spectrum, a spectrogram or a PSD: for a spectrum held as
// consecutive (real, imaginary) float64 pairs, the squared magnitude of each bin times
// a scale. With n = len(dst) and c holding at least n pairs:
//
//	dst[k] = (c[2k]*c[2k] + c[2k+1]*c[2k+1]) * s        k < n
//
// A loop in Go does it a bin at a time with a stride-2 read; a kernel deinterleaves
// four or two pairs per step.
//
// Contract (go-fft#121, the fleet rule: a stated bound, not bit identity): the two
// squares and their sum are non-negative terms, so the result has a relative error of
// at most 4 units of the last place (2^-53) of the exact value: the portable loop
// rounds the two products, the sum and the scale (4 roundings), a kernel fuses the
// second square with the sum (3), and the tests assert the bound against a 200-bit
// computation for every length from 1 to 4,097 and for the two together.
//
// dst must not overlap c, not even exactly: checked, panics with a message saying
// so. len(c) must be at least 2*len(dst); c is not read beyond 2*len(dst).

// SquaredMagnitude writes dst[k] = (re_k² + im_k²)·s for the n = len(dst) complex
// numbers (re_k, im_k) = (c[2k], c[2k+1]).
func SquaredMagnitude(dst, c []float64, s float64) {
	n := len(dst)
	if len(c) < 2*n {
		panic("floats: SquaredMagnitude: len(c) < 2*len(dst)")
	}
	if n == 0 {
		return
	}
	c = c[:2*n]
	if overlaps(dst, c) {
		panic("floats: SquaredMagnitude: dst overlaps c")
	}
	squaredMagnitude(dst, c, s)
}

// sqmagTail is the portable loop from bin from on, and the reference of the kernels:
// every product and sum rounded (the conversions keep the compiler from fusing).
func sqmagTail(dst, c []float64, s float64, from int) {
	for k := from; k < len(dst); k++ {
		re, im := c[2*k], c[2*k+1]
		dst[k] = float64(float64(re*re)+float64(im*im)) * s
	}
}
