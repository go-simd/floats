package floats

// # Mirror combiner (the pair pass of a real transform)
//
// The post-pass of a real-input transform computed through a half-length complex one
// (Makhoul's DCT-II and DCT-III, the untangle of a real FFT) combines each number
// z[k] of a spectrum of n complex numbers with its mirror z[n-k]. For complex numbers
// held as consecutive (real, imaginary) float64 pairs, k = 1 .. n-1:
//
//	u = z[k] + conj(z[n-k])
//	v = -i (z[k] - conj(z[n-k]))      = (Im z[k] + Im z[n-k], Re z[n-k] - Re z[k])
//	r[k] = w1[k] u + w2[k] v          (complex products, w1 and w2 as pairs too)
//
// MirrorCombine stores r[k] as complex number k of dst. MirrorCombineSplit stores
// the real parts and the negated imaginary parts of r where the DCT wants them:
//
//	dst[k]    =  Re r[k]
//	dst[2n-k] = -Im r[k]
//
// Entries 0 and n of the spectrum are the caller's (dst[0:2] of MirrorCombine and
// dst[0], dst[n] of MirrorCombineSplit are not written).
//
// Contract: not bit for bit with any other implementation. A kernel fuses the
// multiply-adds of the products; the tests assert, against the portable loop and a
// 300-bit math/big oracle, an error of at most 4 eps (2^-52) of |w1||u| + |w2||v| plus an absolute floor of 64 smallest subnormals (underflow) on each
// part.
//
// z has 2n floats, n >= 1 (an odd length panics); w1 and w2 hold at least 2n floats
// (entries 0 .. n-1 are read); dst holds at least 2n floats and must not overlap z,
// w1 or w2 (checked, panics). Nothing is written if a check fails.

// MirrorCombine writes r[k], k = 1 .. n-1, as complex numbers into dst[2k], dst[2k+1].
func MirrorCombine(dst, z, w1, w2 []float64) {
	n := mirrorCheck("MirrorCombine", dst, z, w1, w2)
	if n < 2 {
		return
	}
	mirrorCombine(dst, z, w1, w2, n)
}

// MirrorCombineSplit writes Re r[k] to dst[k] and -Im r[k] to dst[2n-k], k = 1 .. n-1.
func MirrorCombineSplit(dst, z, w1, w2 []float64) {
	n := mirrorCheck("MirrorCombineSplit", dst, z, w1, w2)
	if n < 2 {
		return
	}
	mirrorCombineSplit(dst, z, w1, w2, n)
}

// mirrorCheck validates every slice before anything is written and returns n.
func mirrorCheck(name string, dst, z, w1, w2 []float64) int {
	if len(z)&1 != 0 {
		panic("floats: " + name + ": odd len(z)")
	}
	n := len(z) / 2
	if len(w1) < len(z) || len(w2) < len(z) {
		panic("floats: " + name + ": len(w1) or len(w2) < len(z)")
	}
	if len(dst) < len(z) {
		panic("floats: " + name + ": len(dst) < len(z)")
	}
	if len(z) == 0 {
		return 0
	}
	dst = dst[:len(z)]
	if overlaps(dst, z) || overlaps(dst, w1[:len(z)]) || overlaps(dst, w2[:len(z)]) {
		panic("floats: " + name + ": dst overlaps an input")
	}
	return n
}

// mirrorAt is r[k], the reference of the kernels.
func mirrorAt(k, n int, z, w1, w2 []float64) (rr, ri float64) {
	ar, ai := z[2*k], z[2*k+1]
	br, bi := z[2*(n-k)], z[2*(n-k)+1]
	ur, ui := ar+br, ai-bi
	vr, vi := ai+bi, br-ar
	rr = w1[2*k]*ur - w1[2*k+1]*ui + w2[2*k]*vr - w2[2*k+1]*vi
	ri = w1[2*k]*ui + w1[2*k+1]*ur + w2[2*k]*vi + w2[2*k+1]*vr
	return rr, ri
}

// mirrorTail is the portable loop from k = from, and the fallback.
func mirrorTail(dst, z, w1, w2 []float64, n, from int) {
	for k := from; k < n; k++ {
		rr, ri := mirrorAt(k, n, z, w1, w2)
		dst[2*k], dst[2*k+1] = rr, ri
	}
}

func mirrorSplitTail(dst, z, w1, w2 []float64, n, from int) {
	for k := from; k < n; k++ {
		rr, ri := mirrorAt(k, n, z, w1, w2)
		dst[k], dst[2*n-k] = rr, -ri
	}
}
