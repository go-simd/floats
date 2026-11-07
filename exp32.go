package floats

import "math"

// expCR is the double-precision exp of Arm's correctly rounded vector expf
// (optimized-routines, math/aarch64/advsimd/cr_expf.c and cr_expf_utils.h,
// MIT), transcribed one operation per line so that a SIMD kernel can repeat
// it bit for bit: every vfmaq/vfmsq is math.FMA.
//
// x/ln2 is rounded to a multiple of 1/64 by adding a shift whose ULP is 2^-6;
// the low 6 bits of the sum index a table of 2^(i/64), and the bits above
// (with the bias 0x3ff folded into the shift) are the exponent of the scale.
// exp(r) - 1 on |r| <= ln2/128 is a degree-5 polynomial. Valid for
// |x| <= 512, which covers every float32 whose exp is neither 0 nor +Inf.
func expCR(x float64) float64 {
	z := math.FMA(x, crInvLn2, crShift)
	n := z - crShift
	r := math.FMA(-n, crLn2Hi, x)
	r = math.FMA(-n, crLn2Lo, r)
	r2 := r * r
	p01 := math.FMA(r, crC1, crC0)
	p23 := math.FMA(r, crC3, crC2)
	p04 := math.FMA(r2, p23, p01)
	y := math.FMA(r2, p04, r)
	u := math.Float64bits(z)
	s := math.Float64frombits(u<<46&0xfff0000000000000 | crMant[u&63])
	return math.FMA(s, y, s)
}

const (
	crShift  = 0x1.800000000ffc0p+46
	crInvLn2 = 0x1.71547652b82fep+0
	crLn2Hi  = 0x1.62e42fefa39efp-1
	crLn2Lo  = 0x1.abc9e3b39803fp-56
	crC0     = 0x1.fffffffffdbcep-2
	crC1     = 0x1.55555555543c2p-3
	crC2     = 0x1.555573c64f2e3p-5
	crC3     = 0x1.111126b4eff73p-7
)

// crMant are the mantissas of 2^(i/64), i = 0..63 (cr_expf_utils.h).
var crMant = [64]uint64{
	0x0000000000000, 0x02c9a3e778061, 0x059b0d3158574, 0x0874518759bc8,
	0x0b5586cf9890f, 0x0e3ec32d3d1a2, 0x11301d0125b51, 0x1429aaea92de0,
	0x172b83c7d517b, 0x1a35beb6fcb75, 0x1d4873168b9aa, 0x2063b88628cd6,
	0x2387a6e756238, 0x26b4565e27cdd, 0x29e9df51fdee1, 0x2d285a6e4030b,
	0x306fe0a31b715, 0x33c08b26416ff, 0x371a7373aa9cb, 0x3a7db34e59ff7,
	0x3dea64c123422, 0x4160a21f72e2a, 0x44e086061892d, 0x486a2b5c13cd0,
	0x4bfdad5362a27, 0x4f9b2769d2ca7, 0x5342b569d4f82, 0x56f4736b527da,
	0x5ab07dd485429, 0x5e76f15ad2148, 0x6247eb03a5585, 0x6623882552225,
	0x6a09e667f3bcd, 0x6dfb23c651a2f, 0x71f75e8ec5f74, 0x75feb564267c9,
	0x7a11473eb0187, 0x7e2f336cf4e62, 0x82589994cce13, 0x868d99b4492ed,
	0x8ace5422aa0db, 0x8f1ae99157736, 0x93737b0cdc5e5, 0x97d829fde4e50,
	0x9c49182a3f090, 0xa0c667b5de565, 0xa5503b23e255d, 0xa9e6b5579fdbf,
	0xae89f995ad3ad, 0xb33a2b84f15fb, 0xb7f76f2fb5e47, 0xbcc1e904bc1d2,
	0xc199bdd85529c, 0xc67f12e57d14b, 0xcb720dcef9069, 0xd072d4a07897c,
	0xd5818dcfba487, 0xda9e603db3285, 0xdfc97337b9b5f, 0xe502ee78b3ff6,
	0xea4afa2a490da, 0xefa1bee615a27, 0xf50765b6e4540, 0xfa7c1819e90d8,
}

// expCRSlice writes expCR(src[i]) into dst[i] for |src[i]| <= 512, and the
// limits outside: +Inf above, 0 below, NaN for NaN (the float32 results
// there are the same).
func expCRSlice(dst, src []float64) {
	dst = dst[:len(src)]
	for i, x := range src {
		switch {
		case math.Abs(x) <= 512:
			dst[i] = expCR(x)
		case x > 0:
			dst[i] = math.Inf(1)
		case x < 0:
			dst[i] = 0
		default:
			dst[i] = x // NaN
		}
	}
}

const via64Chunk = 256

// Float32Exp writes the correctly rounded float32 exp(src[i]) into dst[i] for
// i < len(src) (every float32 input rounds as the exact value does: proven over
// all 2^32 by TestExp32CorrectlyRounded, which is env-gated). Both operations
// are computed in float64 and rounded once: expCR, Arm's correctly rounded expf
// (optimized-routines, MIT), on an AVX2+FMA kernel on amd64 and a NEON kernel on
// arm64, in Go elsewhere. Nothing is allocated. Moved here from go-ndarray's
// internal/kernels with the AVX2 kernel of go-ndarray#94 and the NEON kernel of
// go-ndarray#95. It panics if len(dst) < len(src).
func Float32Exp(dst, src []float32) {
	if len(dst) < len(src) {
		panic("floats: Float32Exp: len(dst) < len(src)")
	}
	exp32Fast(dst[:len(src)], src)
}

// exp32CR is float32 exp on expCR, rounded once.
func exp32CR(dst, src []float32) {
	src = src[:len(dst)]
	var buf [via64Chunk]float64
	for i := 0; i < len(dst); i += via64Chunk {
		b := buf[:min(via64Chunk, len(dst)-i)]
		for j := range b {
			b[j] = float64(src[i+j])
		}
		expCRSlice(b, b)
		for j, v := range b {
			dst[i+j] = float32(v)
		}
	}
}
