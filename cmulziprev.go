package floats

// # Mirrored-halves gather with a complex multiply (the pre-pass of a DCT-III)
//
// The input of a DCT-III or of the inverse of a real transform holds, in two
// contiguous runs, the real and the imaginary parts of the numbers the half-length
// complex transform wants: the real part of number j is x[j] and its imaginary part
// is read from the other end, y[m-1-j]. Makhoul's DCT-III (go-fft/fft#162) is this
// pass, a complex multiply by a table, then MirrorCombine. With m = len(dst)/2
// complex numbers (consecutive (real, imaginary) float64 pairs) and w holding m of
// them, for j < m:
//
//	dst[j] = w[j] * (x[j] - i y[m-1-j])
//
// that is
//
//	dst[2j]   = w[2j]*x[j] + w[2j+1]*y[m-1-j]
//	dst[2j+1] = w[2j+1]*x[j] - w[2j]*y[m-1-j]
//
// ConjReverse reverses and conjugates complex numbers and CMulGather gathers an even
// run with a reversed odd run (stride 2); neither reads two contiguous runs, one of
// them reversed.
//
// Contract (a stated bound, go-fft#121): each part is within 4 eps (2^-52) of
// |w|·|(x, y')| plus an absolute floor of 64 smallest subnormals (underflow) of the
// exact value, asserted against a 300-bit math/big oracle. A kernel fuses the products
// and the portable loop does not, so they can differ in the last bit.
//
// len(dst) must be even, len(w) >= len(dst), len(x) and len(y) >= len(dst)/2; dst must
// not overlap x, y or w (checked, panics; x and y may overlap each other, nothing is
// written to them). Nothing is written if a check fails.

// CMulZipRev writes dst[j] = w[j] * (x[j] - i y[m-1-j]) for j < m = len(dst)/2.
func CMulZipRev(dst, x, y, w []float64) {
	n := len(dst)
	if n&1 != 0 {
		panic("floats: CMulZipRev: odd len(dst)")
	}
	m := n / 2
	if len(x) < m || len(y) < m || len(w) < n {
		panic("floats: CMulZipRev: len(x) or len(y) < len(dst)/2, or len(w) < len(dst)")
	}
	if n == 0 {
		return
	}
	x, y, w = x[:m], y[:m], w[:n]
	if overlaps(dst, x) || overlaps(dst, y) || overlaps(dst, w) {
		panic("floats: CMulZipRev: dst overlaps an input")
	}
	cmulZipRev(dst, x, y, w)
}

// zipRevTail is the portable loop from j = from, and the reference of the kernel.
func zipRevTail(dst, x, y, w []float64, from int) {
	m := len(dst) / 2
	for j := from; j < m; j++ {
		wr, wi, a, b := w[2*j], w[2*j+1], x[j], y[m-1-j]
		dst[2*j] = wr*a + wi*b
		dst[2*j+1] = wi*a - wr*b
	}
}
