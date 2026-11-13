package floats

// # Twiddle passes of the DCT-IV and the MDCT (complex multiply by a table, with
// an even/reversed-odd gather or scatter)
//
// A DCT-IV of even length n is one n/2-point complex FFT between two passes over
// the data (go-fft/fft dct.go, Makhoul 1980; FFTW's REDFT11 does the same): a
// pre-twiddle that reads the real input as complex pairs of its even samples and
// its odd samples taken from the end, and multiplies them by a table, and a
// post-twiddle that multiplies the FFT's output by a table and scatters it back,
// real parts to the even positions and the negated imaginary parts to the odd
// positions from the end. Those two passes are more than half of a small DCT-IV
// or MDCT (go-fft/fft#132: 45 of 103 ns at 64 points, 750 of 1,300 ns at 1,024)
// and are the shape SIMD does well and the compiler does not vectorise: a
// stride-2 deinterleave, a lane reversal, a complex multiply, an interleave.
//
// With h = len(tw)/2 and n = 2h = len(tw), a complex number is two consecutive
// float64 (real, imaginary), tw holds h of them, and for m < h:
//
//	CMulGather:   dst[2m]   = x[2m]*tw[2m]   - x[n-1-2m]*tw[2m+1]
//	              dst[2m+1] = x[2m]*tw[2m+1] + x[n-1-2m]*tw[2m]
//
//	CMulScatter:  zr = tw[2m]*c[2m] - tw[2m+1]*c[2m+1]
//	              zi = tw[2m]*c[2m+1] + tw[2m+1]*c[2m]
//	              dst[2m]     = f*zr
//	              dst[n-1-2m] = -f*zi
//
// CMulGatherFrom32 reads x as float32 (widened to float64 in the kernel, so the
// arithmetic is float64) and CMulScatterTo32 writes dst as float32 (each output
// rounded once to float32 after the multiply by f): the float32-data,
// float64-arithmetic pairing of go-fft/fft's DCTMixed and MDCTMixed, without a
// conversion pass.
//
// Contract (go-fft#121, the fleet rule: a stated bound, not bit identity): each
// output is the exact value of its formula on the inputs, rounded with an error of
// at most 4 units of the last place (2^-53 for float64 outputs) of the sum of the
// magnitudes of its two products, times |f| for the scatter, and, for a float32
// output, plus half a unit of the last place of float32 of the output. A kernel
// may fuse a product with the sum (one rounding fewer) and the portable code does
// not, so the two can differ in the last bit; both are within the bound, which the
// tests assert against a 200-bit computation.
//
// dst must not overlap the other operands. n must be even; len(x), len(c) and
// len(dst) must be at least n, and len(tw) = n. The slices are not read or
// written beyond n. A panic otherwise.

// CMulGather writes the pre-twiddle of a DCT-IV: see the formulas above.
func CMulGather(dst, x, tw []float64) {
	n := len(tw)
	if n&1 != 0 || len(x) < n || len(dst) < n {
		panic("floats: CMulGather: len(tw) is odd, or len(x) or len(dst) < len(tw)")
	}
	if n == 0 {
		return
	}
	cmulGather(dst[:n], x[:n], tw)
}

// CMulGatherFrom32 is CMulGather with x read as float32 and widened.
func CMulGatherFrom32(dst []float64, x []float32, tw []float64) {
	n := len(tw)
	if n&1 != 0 || len(x) < n || len(dst) < n {
		panic("floats: CMulGatherFrom32: len(tw) is odd, or len(x) or len(dst) < len(tw)")
	}
	if n == 0 {
		return
	}
	cmulGather32(dst[:n], x[:n], tw)
}

// CMulScatter writes the post-twiddle of a DCT-IV: see the formulas above.
func CMulScatter(dst, c, tw []float64, f float64) {
	n := len(tw)
	if n&1 != 0 || len(c) < n || len(dst) < n {
		panic("floats: CMulScatter: len(tw) is odd, or len(c) or len(dst) < len(tw)")
	}
	if n == 0 {
		return
	}
	cmulScatter(dst[:n], c[:n], tw, f)
}

// CMulScatterTo32 is CMulScatter with each output rounded to float32.
func CMulScatterTo32(dst []float32, c, tw []float64, f float64) {
	n := len(tw)
	if n&1 != 0 || len(c) < n || len(dst) < n {
		panic("floats: CMulScatterTo32: len(tw) is odd, or len(c) or len(dst) < len(tw)")
	}
	if n == 0 {
		return
	}
	cmulScatter32(dst[:n], c[:n], tw, f)
}

// cmulGatherTail is the portable loop from complex index from on, and the
// reference of the kernels: every product and every sum rounded (the conversions
// keep the compiler from fusing).
func cmulGatherTail(dst, x, tw []float64, from int) {
	n := len(tw)
	for m := from; m < n/2; m++ {
		e, o := x[2*m], x[n-1-2*m]
		wr, wi := tw[2*m], tw[2*m+1]
		dst[2*m] = float64(e*wr) - float64(o*wi)
		dst[2*m+1] = float64(e*wi) + float64(o*wr)
	}
}

func cmulGather32Tail(dst []float64, x []float32, tw []float64, from int) {
	n := len(tw)
	for m := from; m < n/2; m++ {
		e, o := float64(x[2*m]), float64(x[n-1-2*m])
		wr, wi := tw[2*m], tw[2*m+1]
		dst[2*m] = float64(e*wr) - float64(o*wi)
		dst[2*m+1] = float64(e*wi) + float64(o*wr)
	}
}

func cmulScatterRef(dst, c, tw []float64, f float64) {
	n := len(tw)
	for m := 0; m < n/2; m++ {
		cr, ci := c[2*m], c[2*m+1]
		wr, wi := tw[2*m], tw[2*m+1]
		zr := float64(wr*cr) - float64(wi*ci)
		zi := float64(wr*ci) + float64(wi*cr)
		dst[2*m] = float64(zr * f)
		dst[n-1-2*m] = -float64(zi * f)
	}
}

func cmulScatter32Ref(dst []float32, c, tw []float64, f float64) {
	n := len(tw)
	for m := 0; m < n/2; m++ {
		cr, ci := c[2*m], c[2*m+1]
		wr, wi := tw[2*m], tw[2*m+1]
		zr := float64(wr*cr) - float64(wi*ci)
		zi := float64(wr*ci) + float64(wi*cr)
		dst[2*m] = float32(float64(zr * f))
		dst[n-1-2*m] = float32(-float64(zi * f))
	}
}
