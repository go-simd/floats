package floats

// # Conjugate-reverse of complex numbers (the mirror of a Hermitian spectrum)
//
// The conjugate mirror that turns the N/2+1 bins of a real transform into the full
// spectrum, X[N-k] = conj(X[k]): for complex numbers held as consecutive (real,
// imaginary) float64 pairs, the same numbers in reverse order with the imaginary parts
// negated. With n = len(dst)/2 complex numbers:
//
//	dst[2k]   =  src[2(n-1-k)]
//	dst[2k+1] = -src[2(n-1-k)+1]            k < n
//
// A loop in Go does it a number at a time; a kernel reverses two or four numbers per
// step (a lane swap, then a sign flip of the imaginary lanes).
//
// Contract: every output is a copy or a negation of one input, so the result is exact
// (not within a bound: equal bit for bit to the portable loop, which the tests assert
// for every length from 0 to 4,097, signed zeros and NaN payloads included).
//
// dst must not overlap src, not even exactly: checked, panics with a message saying
// so. len(dst) must be even and len(src) at least len(dst); src is not read beyond
// len(dst).

// ConjReverse writes the n = len(dst)/2 complex numbers (src[2k], src[2k+1]) conjugated
// and in reverse order into dst.
func ConjReverse(dst, src []float64) {
	if len(dst)&1 != 0 {
		panic("floats: ConjReverse: odd len(dst)")
	}
	if len(src) < len(dst) {
		panic("floats: ConjReverse: len(src) < len(dst)")
	}
	if len(dst) == 0 {
		return
	}
	src = src[:len(dst)]
	if overlaps(dst, src) {
		panic("floats: ConjReverse: dst overlaps src")
	}
	conjReverse(dst, src)
}

// conjRevTail is the portable loop from complex number from on, and the reference of
// the kernels.
func conjRevTail(dst, src []float64, from int) {
	n := len(dst) / 2
	for k := from; k < n; k++ {
		j := 2 * (n - 1 - k)
		dst[2*k], dst[2*k+1] = src[j], -src[j+1]
	}
}
