package floats

// # Fold and unfold of the MDCT (and the IMDCT)
//
// An MDCT of N = 4h samples is a DCT-IV of M = 2h values on a folded copy of the
// block, and an IMDCT is a DCT-IV followed by the inverse arrangement of its M
// outputs into N samples (Wikipedia, "Modified discrete cosine transform", the
// fold (a, b, c, d) -> (-c_r - d, a - b_r) of the four quarters of the block, r the
// reversal). Every MDCT codec (AAC, Vorbis, Opus CELT, AC-3, MP3) does both. With
// m = M = N/2 and i < h, the fold of x (len N) into u (len M) and the unfold of u
// into dst (len N) are
//
//	MDCTFold:    u[i]   = -(x[m+h-1-i] + x[m+h+i])
//	             u[h+i] =   x[i] - x[m-1-i]
//
//	MDCTUnfold:  dst[i]       =  u[h+i]
//	             dst[m-1-i]   = -u[h+i]
//	             dst[m+h-1-i] = -u[i]
//	             dst[m+h+i]   = -u[i]
//
// Both read and write forward and reversed runs of consecutive elements, which a
// loop in Go does an element at a time and a SIMD kernel does with a lane reversal.
// The float32 forms keep the data float32 and the arithmetic float64: MDCTFoldFrom32
// widens x while it adds (the sum of two float32 in float64), MDCTUnfoldTo32 rounds
// each output to float32 (a negation is exact, so the output is the float32 of u's
// value or its opposite).
//
// Contract: the fold is two float64 additions or subtractions per output and the
// unfold moves and negates, none fused, so every kernel gives exactly the result of
// the portable loop (bit for bit, up to the sign of a zero, which the tests treat as
// equal); the bound is the rounding of one addition, half a unit of the last place
// of the exact result, and the tests assert the equality with the portable loop and
// the bound against a 200-bit computation.
//
// dst must not overlap x or u, not even exactly: the kernels read and write both
// ends. The overlap is checked and panics with a message saying so. len(x) must be a
// multiple of 4 and len(dst) at least len(x)/2 for the fold; len(u) must be even and
// len(dst) at least 2*len(u) for the unfold; the slices are not read or written
// beyond those lengths. A panic otherwise.

// MDCTFold writes the fold of the N = len(x) samples of x into dst[:N/2].
func MDCTFold(dst, x []float64) {
	n := len(x)
	if n&3 != 0 || len(dst) < n/2 {
		panic("floats: MDCTFold: len(x) is not a multiple of 4, or len(dst) < len(x)/2")
	}
	if n == 0 {
		return
	}
	dst = dst[:n/2]
	if overlaps(dst, x) {
		panic("floats: MDCTFold: dst overlaps x")
	}
	mdctFold(dst, x)
}

// MDCTFoldFrom32 is MDCTFold with x read as float32 and widened.
func MDCTFoldFrom32(dst []float64, x []float32) {
	n := len(x)
	if n&3 != 0 || len(dst) < n/2 {
		panic("floats: MDCTFoldFrom32: len(x) is not a multiple of 4, or len(dst) < len(x)/2")
	}
	if n == 0 {
		return
	}
	dst = dst[:n/2]
	if overlaps(dst, x) {
		panic("floats: MDCTFoldFrom32: dst overlaps x")
	}
	mdctFold32(dst, x)
}

// MDCTUnfold writes the unfolding of the M = len(u) values of u into dst[:2M].
func MDCTUnfold(dst, u []float64) {
	m := len(u)
	if m&1 != 0 || len(dst) < 2*m {
		panic("floats: MDCTUnfold: len(u) is odd, or len(dst) < 2*len(u)")
	}
	if m == 0 {
		return
	}
	dst = dst[:2*m]
	if overlaps(dst, u) {
		panic("floats: MDCTUnfold: dst overlaps u")
	}
	mdctUnfold(dst, u)
}

// MDCTUnfoldTo32 is MDCTUnfold with each output rounded to float32.
func MDCTUnfoldTo32(dst []float32, u []float64) {
	m := len(u)
	if m&1 != 0 || len(dst) < 2*m {
		panic("floats: MDCTUnfoldTo32: len(u) is odd, or len(dst) < 2*len(u)")
	}
	if m == 0 {
		return
	}
	dst = dst[:2*m]
	if overlaps(dst, u) {
		panic("floats: MDCTUnfoldTo32: dst overlaps u")
	}
	mdctUnfold32(dst, u)
}

// foldTail is the portable fold from index from on, and the reference of the kernels.
func foldTail(dst, x []float64, from int) {
	n := len(x)
	m, h := n/2, n/4
	for i := from; i < h; i++ {
		dst[i] = -(x[m+h-1-i] + x[m+h+i])
		dst[h+i] = x[i] - x[m-1-i]
	}
}

func fold32Tail(dst []float64, x []float32, from int) {
	n := len(x)
	m, h := n/2, n/4
	for i := from; i < h; i++ {
		dst[i] = -(float64(x[m+h-1-i]) + float64(x[m+h+i]))
		dst[h+i] = float64(x[i]) - float64(x[m-1-i])
	}
}

func unfoldTail(dst, u []float64, from int) {
	m := len(u)
	h := m / 2
	for i := from; i < h; i++ {
		s, t := u[i], u[h+i]
		dst[i] = t
		dst[m-1-i] = -t
		dst[m+h-1-i] = -s
		dst[m+h+i] = -s
	}
}

func unfold32Tail(dst []float32, u []float64, from int) {
	m := len(u)
	h := m / 2
	for i := from; i < h; i++ {
		s, t := float32(u[i]), float32(u[h+i])
		dst[i] = t
		dst[m-1-i] = -t
		dst[m+h-1-i] = -s
		dst[m+h+i] = -s
	}
}
