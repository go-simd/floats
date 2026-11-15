package floats

import "math"

// # Elementwise Abs, Neg, Square, Floor, Ceil, Trunc and RoundEven
//
// The float64 elementwise operations go-ndarray still ran as scalar Go loops
// (go-ndarray/ndarray#118, cause 2). They are exact operations: every result is
// the value of the Go expression in its doc comment, bit for bit, on every input
// and on every architecture and CPU feature level, because each element is one
// IEEE-754 operation that has no rounding choice (Abs, Neg: a sign-bit
// operation; Floor, Ceil, Trunc, RoundEven: roundToIntegral, always exact;
// Square: one multiplication, one rounding). Like Add and Sqrt they work on the
// first len(src) elements, panic if len(dst) < len(src), and allow dst to be the
// very same slice as src (partially overlapping slices are not supported).
//
// NaN: Abs and Neg only touch the sign bit, so a NaN keeps its payload and
// quiet bit whatever its sign. The others return a NaN for a NaN: a quiet NaN
// keeps its sign and payload on every target but riscv64 (whose floating point
// arithmetic produces the canonical NaN), a signalling NaN comes back quieted
// where the hardware quiets it (Go's own math.Floor does the same on the targets
// where it is an instruction), exactly like the Go expression run on the same
// machine.
//
// Kernels: amd64 AVX2 (sign masks with VANDPD/VXORPD, VMULPD, VROUNDPD with
// imm8 9, 10, 11, 8), the SSE2 baseline for Abs, Neg and Square (the rounding
// functions stay scalar there: SSE2 has no ROUNDPD); arm64 NEON (FABS, FNEG,
// FMUL, FRINTM, FRINTP, FRINTZ, FRINTN); ppc64le VSX (xvabsdp, xvnegdp, xvmuldp,
// xvrdpim, xvrdpip, xvrdpiz, xvrdpic); s390x vector facility (VFLPDB, VFLCDB,
// VFMDB, VFIDB); loong64 LASX (bit operations, XVMULD, XVFRINTRM, XVFRINTRP,
// XVFRINTRZ, XVFRINTRNE); riscv64 RVV (VFSGNJX, VFSGNJN, VFMUL; the rounding
// functions stay scalar, RVV 1.0 has no vector round-to-integral). Every kernel
// does the whole blocks of eight elements and the Go code finishes the last
// seven with the scalar expression.

// Abs computes dst[i] = math.Abs(src[i]) for i < len(src). It panics if
// len(dst) < len(src).
func Abs(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Abs: len(dst) < len(src)")
	}
	vabs(dst[:len(src)], src)
}

// Neg computes dst[i] = -src[i] for i < len(src): the sign bit is flipped, also
// for zeros, infinities and NaNs. It panics if len(dst) < len(src).
func Neg(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Neg: len(dst) < len(src)")
	}
	vneg(dst[:len(src)], src)
}

// Square computes dst[i] = src[i] * src[i] for i < len(src), one rounding per
// element (np.square). It panics if len(dst) < len(src).
func Square(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Square: len(dst) < len(src)")
	}
	vsquare(dst[:len(src)], src)
}

// Floor computes dst[i] = math.Floor(src[i]) for i < len(src): the largest
// integer not above the value, ±0, ±Inf and NaN unchanged (np.floor). It panics
// if len(dst) < len(src).
func Floor(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Floor: len(dst) < len(src)")
	}
	vfloor(dst[:len(src)], src)
}

// Ceil computes dst[i] = math.Ceil(src[i]) for i < len(src): the smallest
// integer not below the value; Ceil(-0.5) is -0 (np.ceil). It panics if
// len(dst) < len(src).
func Ceil(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Ceil: len(dst) < len(src)")
	}
	vceil(dst[:len(src)], src)
}

// Trunc computes dst[i] = math.Trunc(src[i]) for i < len(src): the integer part,
// rounding toward zero, with the sign of the value (Trunc(-0.5) is -0)
// (np.trunc). It panics if len(dst) < len(src).
func Trunc(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: Trunc: len(dst) < len(src)")
	}
	vtrunc(dst[:len(src)], src)
}

// RoundEven computes dst[i] = math.RoundToEven(src[i]) for i < len(src): the
// nearest integer, halfway cases to the even one (np.rint, and np.round with no
// decimals). It panics if len(dst) < len(src).
func RoundEven(dst, src []float64) {
	if len(dst) < len(src) {
		panic("floats: RoundEven: len(dst) < len(src)")
	}
	vrint(dst[:len(src)], src)
}

// ---- portable reference kernels (the oracle of the tests, and the kernel on
// every architecture without one, and for the last elements of every call). The
// callers have sliced dst to len(src).

func absRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Abs(v)
	}
}

func negRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = -v
	}
}

func squareRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = v * v
	}
}

func floorRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Floor(v)
	}
}

func ceilRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Ceil(v)
	}
}

func truncRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.Trunc(v)
	}
}

func rintRef(dst, src []float64) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = math.RoundToEven(v)
	}
}
