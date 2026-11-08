package floats

// # Horner and Clenshaw evaluation over an array of points, bit for bit as NumPy
//
// Polyval and Chebval evaluate ONE series at MANY points. They reproduce, per
// point and operation by operation, what numpy.polynomial 2.5.4 does:
//
//	polynomial/polynomial.py, polyval:
//	    c0 = c[-1] + x*0
//	    for i in range(2, len(c)+1): c0 = c[-i] + c0*x
//
//	polynomial/chebyshev.py, chebval:
//	    len(c) == 1: c0 = c[0]; c1 = 0
//	    len(c) == 2: c0 = c[0]; c1 = c[1]
//	    otherwise:   x2 = 2*x; c0 = c[-2]; c1 = c[-1]
//	                 for i in range(3, len(c)+1): tmp = c0; c0 = c[-i] - c1; c1 = tmp + c1*x2
//	    return c0 + c1*x
//
// NumPy rounds every multiply and add on its own (ufunc temporaries), so nothing
// here is fused: VMULPD+VADDPD on amd64 (AVX2, or SSE2 without AVX2+FMA), VFMUL+
// VFADD on arm64, and portable loops that write each product through an explicit
// float64 conversion (which keeps the Go compiler from fusing it) on every other
// architecture (floats#30). The SIMD lanes run over different points, so each
// point sees exactly the scalar order above; the results are bit-identical to the
// loops and to NumPy (the oracle tests compare the bits), including Inf and NaN
// inputs (c[-1] + x*0 is NaN for x = ±Inf, as in NumPy), apart from the payload of
// a NaN that an operation itself creates, which is left to the hardware as in NumPy.
//
// dst may be the very same slice as x (partially overlapping slices are not
// supported); c must not overlap dst.

// Polyval computes dst[i] = p(x[i]) for i < len(x), where p is the polynomial
// c[0] + c[1]*t + ... + c[len(c)-1]*t^(len(c)-1) evaluated by Horner's rule in
// NumPy's order (numpy.polynomial.polynomial.polyval). It panics if len(dst) <
// len(x) or len(c) == 0 (NumPy raises IndexError). Elements of dst past len(x)
// are left untouched.
func Polyval(dst, x, c []float64) {
	if len(dst) < len(x) {
		panic("floats: Polyval: len(dst) < len(x)")
	}
	if len(c) == 0 {
		panic("floats: Polyval: no coefficients")
	}
	if len(x) == 0 {
		return
	}
	polyvalKernel(dst[:len(x)], x, c)
}

// Chebval computes dst[i] = sum c[k]*T_k(x[i]) for i < len(x) by Clenshaw's
// recurrence in NumPy's order (numpy.polynomial.chebyshev.chebval). It panics if
// len(dst) < len(x) or len(c) == 0. Elements of dst past len(x) are left untouched.
func Chebval(dst, x, c []float64) {
	if len(dst) < len(x) {
		panic("floats: Chebval: len(dst) < len(x)")
	}
	if len(c) == 0 {
		panic("floats: Chebval: no coefficients")
	}
	if len(x) == 0 {
		return
	}
	if len(c) == 1 { // c0 + 0*x: no recurrence, no kernel
		chebvalRef(dst[:len(x)], x, c)
		return
	}
	chebvalKernel(dst[:len(x)], x, c)
}

// ---- portable reference kernels ----
//
// The oracle the tests compare the SIMD kernels and the NumPy fixtures against,
// and the kernel on every architecture without one.

func polyvalRef(dst, x, c []float64) {
	dst = dst[:len(x)]
	n := len(c)
	for i, v := range x {
		c0 := c[n-1] + float64(v*0)
		for j := n - 2; j >= 0; j-- {
			c0 = c[j] + float64(c0*v)
		}
		dst[i] = c0
	}
}

func chebvalRef(dst, x, c []float64) {
	dst = dst[:len(x)]
	n := len(c)
	for i, v := range x {
		var c0, c1 float64
		switch n {
		case 1:
			c0, c1 = c[0], 0
		case 2:
			c0, c1 = c[0], c[1]
		default:
			x2 := v + v
			c0, c1 = c[n-2], c[n-1]
			for j := n - 3; j >= 0; j-- {
				tmp := c0
				c0 = c[j] - c1
				c1 = tmp + float64(c1*x2)
			}
		}
		dst[i] = c0 + float64(c1*v)
	}
}
