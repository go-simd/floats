package floats

// # Unfused elementwise kernels, bit for bit as NumPy evaluates them
//
// Axpy, Xpby and Axpby3 are FUSED: one rounding per element, VFMADD on amd64,
// VFMLA on arm64. NumPy never fuses: the expression y + alpha*x is two ufunc
// calls, a multiply that rounds into a temporary and an add that rounds again,
// and a solver validated by exact equality against SciPy has to take that same
// path. The NoFMA kernels below do exactly that: ONE IEEE 754 rounding per
// binary operation, in the order NumPy evaluates the temporaries (Python
// evaluates a + F*(b - c) as t1 = b - c; t2 = F*t1; a + t2, and
// alpha*a + beta*b + gamma*c from left to right). Each result is bit-identical
// to the NumPy expression (checked against NumPy 2.5.4 fixtures) and to the
// portable loops below on every architecture; only the payload of a NaN that an
// operation itself creates (Inf - Inf, 0*Inf) is left to the hardware, as in
// NumPy.
//
// The vector kernels are plain multiplies, adds and subtracts (VMULPD/VADDPD/
// VSUBPD on amd64 with AVX2, VFMUL/VFADD/VFSUB on arm64), never a fused
// instruction. The portable code writes every product through an explicit
// float64 conversion, which the Go specification says prevents the compiler from
// fusing x*y + z into an FMA (it otherwise may, on arm64, ppc64le, s390x, riscv64
// and loong64).
//
// Like the fused updates, they work on the first len(x) (len(a)) elements, panic
// if another slice is shorter, and allow the output to be the very same slice as
// an input (partially overlapping slices are not supported).

// AxpyNoFMA computes y[i] = y[i] + alpha*x[i] for i < len(x) with two roundings,
// the product first: NumPy's y + alpha*x. Elements of y past len(x) are left
// untouched. It panics if len(y) < len(x).
func AxpyNoFMA(alpha float64, x, y []float64) {
	if len(y) < len(x) {
		panic("floats: AxpyNoFMA: len(y) < len(x)")
	}
	axpyNoFMA(alpha, x, y[:len(x)])
}

// XpbyNoFMA computes y[i] = x[i] + beta*y[i] for i < len(x) with two roundings,
// the product first: NumPy's x + beta*y. It panics if len(y) < len(x).
func XpbyNoFMA(x []float64, beta float64, y []float64) {
	if len(y) < len(x) {
		panic("floats: XpbyNoFMA: len(y) < len(x)")
	}
	xpbyNoFMA(beta, x, y[:len(x)])
}

// Axpby3NoFMA computes dst[i] = alpha*a[i] + beta*b[i] + gamma*c[i] for
// i < len(a), unfused and left to right, as NumPy evaluates
// alpha*a + beta*b + gamma*c: ((alpha*a[i]) + (beta*b[i])) + (gamma*c[i]), five
// roundings. It is the unfused counterpart of Axpby3. It panics if len(b),
// len(c) or len(dst) is less than len(a).
func Axpby3NoFMA(dst []float64, alpha float64, a []float64, beta float64, b []float64, gamma float64, c []float64) {
	if len(b) < len(a) || len(c) < len(a) || len(dst) < len(a) {
		panic("floats: Axpby3NoFMA: len(b), len(c) or len(dst) < len(a)")
	}
	n := len(a)
	axpby3NoFMA(dst[:n], alpha, a, beta, b[:n], gamma, c[:n])
}

// AxpyDiffNoFMA computes dst[i] = a[i] + f*(b[i] - c[i]) for i < len(a): the
// difference, the product and the sum each rounded in turn, as NumPy evaluates
// a + F*(b - c). It is the mutation step of differential evolution (rand/1,
// best/1 and the second half of rand-to-best/1; for b += F*(x - b) pass
// dst = a = c = b). It panics if len(b), len(c) or len(dst) is less than len(a).
func AxpyDiffNoFMA(dst, a []float64, f float64, b, c []float64) {
	if len(b) < len(a) || len(c) < len(a) || len(dst) < len(a) {
		panic("floats: AxpyDiffNoFMA: len(b), len(c) or len(dst) < len(a)")
	}
	n := len(a)
	axpyDiffNoFMA(dst[:n], a, f, b[:n], c[:n])
}

// ---- portable reference kernels ----
//
// The oracle the tests compare the SIMD kernels against, and the kernel on every
// architecture without one. The float64(...) conversion around each product is
// what keeps the compiler from fusing it with the following add.

func axpyNoFMARef(alpha float64, x, y []float64) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] = y[i] + float64(alpha*v)
	}
}

func xpbyNoFMARef(beta float64, x, y []float64) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] = v + float64(beta*y[i])
	}
}

func axpby3NoFMARef(dst []float64, alpha float64, a []float64, beta float64, b []float64, gamma float64, c []float64) {
	b, c, dst = b[:len(a)], c[:len(a)], dst[:len(a)]
	for i, v := range a {
		t := float64(alpha*v) + float64(beta*b[i])
		dst[i] = t + float64(gamma*c[i])
	}
}

func axpyDiffNoFMARef(dst, a []float64, f float64, b, c []float64) {
	b, c, dst = b[:len(a)], c[:len(a)], dst[:len(a)]
	for i, v := range a {
		d := b[i] - c[i]
		dst[i] = v + float64(f*d)
	}
}

// AxpyRowsNoFMA applies len(a) unfused AXPY updates to y in one pass: for
// j < len(y),
//
//	y[j] = (((y[j] + a[0]*x[0*ldx+j]) + a[1]*x[1*ldx+j]) + ...) + a[r-1]*x[(r-1)*ldx+j]
//
// with r = len(a), every product rounded before its addition and the rows added
// in increasing order. The result is bit-identical to r successive
// AxpyNoFMA(a[k], x[k*ldx:k*ldx+len(y)], y) calls, but a block of y stays in
// registers across all r rows, so y is loaded and stored once instead of r times.
// It is the row of an exact matrix product C = A*B in i-k-j order: pass
// a = A[i,:], x = B (row-major, ldx = its row stride), y = C[i,:], and the entry
// is the left-to-right sum in increasing k that NumPy's matmul loop (and any
// reference that does not fuse) computes.
//
// x holds row k at x[k*ldx:]; it must not overlap y. It panics if ldx < 0 or if
// x is shorter than (len(a)-1)*ldx + len(y). With len(a) == 0 or len(y) == 0 it
// does nothing (and checks nothing).
func AxpyRowsNoFMA(a []float64, x []float64, ldx int, y []float64) {
	r, n := len(a), len(y)
	if r == 0 || n == 0 {
		return
	}
	if ldx < 0 {
		panic("floats: AxpyRowsNoFMA: ldx < 0")
	}
	// (r-1)*ldx + n <= len(x), without overflow on 32-bit ints.
	if len(x) < n || (r > 1 && ldx > (len(x)-n)/(r-1)) {
		panic("floats: AxpyRowsNoFMA: x shorter than (len(a)-1)*ldx + len(y)")
	}
	axpyRowsNoFMA(a, x, ldx, y)
}

func axpyRowsNoFMARef(a []float64, x []float64, ldx int, y []float64) {
	for j := range y {
		s := y[j]
		for k, ak := range a {
			s = s + float64(ak*x[k*ldx+j])
		}
		y[j] = s
	}
}
