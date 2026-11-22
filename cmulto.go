package floats

// # Pointwise complex product
//
// The product of two vectors of complex numbers, element by element: the chirp and
// the spectral products of Bluestein's algorithm, the filter of a frequency-domain
// convolution, the twiddle pass of a mixed-radix FFT. NumPy's multiply on complex128.
//
//	dst[i] = a[i] * b[i]      for i < len(dst)
//
// Contract (a stated bound, go-fft#121; bit identity with another code is not the
// goal): with a[i] = (ar, ai) and b[i] = (br, bi), each part of dst[i] is within
// 2 eps (eps = 2^-52) of |a[i]|·|b[i]| of the exact product, plus an absolute floor of
// 64 smallest subnormals when a product underflows; the tests assert it against a
// 300-bit math/big oracle. The AVX2+FMA kernel computes the real part as
// fma(ar, br, -(ai*bi)) and the imaginary part as fma(ar, bi, ai*br) (two roundings
// of one product each, one fused); the portable loop is Go's complex product, which a
// compiler may fuse in the same way (arm64 does, amd64 does not), so the paths can
// differ in the last bit and all are within the bound. Overflow gives Inf or NaN as
// Go's complex product does; the bound holds where nothing overflows.
//
// len(a) and len(b) must be at least len(dst). dst may be a or b (in place), exactly:
// any other overlap with a or b is undefined. Nothing is written if the check fails.

// CMulTo writes dst[i] = a[i]*b[i] for i < len(dst). It panics, before writing
// anything, if len(a) or len(b) is less than len(dst).
func CMulTo(dst, a, b []complex128) {
	n := len(dst)
	if len(a) < n || len(b) < n {
		// Slicing a[:n] would not catch it: a shorter slice with room in its
		// capacity reslices without a panic, and the kernel trusts n.
		panic("floats: CMulTo: len(a) or len(b) < len(dst)")
	}
	a, b = a[:n], b[:n]
	cmulTo(dst, a, b)
}

// cmulToTail is the portable loop from i = from, and the reference of the kernels:
// Go's complex product.
func cmulToTail(dst, a, b []complex128, from int) {
	for i := from; i < len(dst); i++ {
		dst[i] = a[i] * b[i]
	}
}
