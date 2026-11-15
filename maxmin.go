package floats

import "math"

// # Pairwise Maximum and Minimum
//
// The float64 elementwise maximum and minimum go-ndarray still ran as scalar Go
// loops (go-ndarray/ndarray#98, "K1 remainder"; #118). The contract is ndarray's
// scalar definition, internal/kernels Maximum and Minimum at v0.44.0, which is the
// Go builtin max / min (not math.Max / math.Min), that is the language's rule
// spelled out by the runtime's fmax / fmin (runtime/minmax.go), and the kernels
// reproduce it bit for bit, payloads included:
//
//   - a NaN in either operand gives a NaN. When only one operand is a NaN the
//     result is that operand, unchanged (sign, payload and quiet bit: a signalling
//     NaN stays signalling). When both are NaNs the result is b[i], the second
//     operand. (The Go compiler's own max may quiet a signalling NaN or pick the
//     other operand of two NaNs on a target whose instruction does that; the rule
//     here is the language's portable one, the same on every target and CPU
//     feature level.)
//   - max(-0, +0) and max(+0, -0) are +0 (the result is -0 only if both are -0),
//     and min(-0, +0) and min(+0, -0) are -0, whatever the operand order: the
//     bitwise AND of the two zeros for Maximum, the bitwise OR for Minimum. (NumPy's
//     np.maximum returns a when a >= b, so it returns the first operand for a
//     (-0, +0) pair; the Go rule does not look at the order.)
//   - ±Inf compare as the extreme finite values do; max(NaN, +Inf) is NaN.
//   - otherwise the larger (smaller) operand; two equal non-zero values are
//     the same bits.
//
// Like Add they work on the first len(a) elements, panic if another slice is
// shorter, and allow dst to be the very same slice as a or b (partially
// overlapping slices are not supported). There is no float32 variant: the
// repository has none for the comparable elementwise kernels (Add, Mul, Sqrt).
//
// Kernels: amd64 AVX2 (VCMPPD, VBLENDVPD, VANDPD / VORPD) and the SSE2 baseline
// (CMPPD, ANDPD / ANDNPD / ORPD); arm64 NEON (FMAX / FMIN for the numbers and the
// zeros, FCMEQ and BSL to put the NaN back); ppc64le VSX (xvcmpgtdp / xvcmpeqdp /
// xxsel); s390x vector facility (VFCHDB, VFCEDB, VSEL); loong64 LASX (XVFCMP,
// XVBITSEL); riscv64 RVV (vmflt, vmfne, vmfeq, vmerge). Every kernel does the whole
// blocks of eight elements and the Go code finishes the last seven with the scalar
// rule. See each maxmin_<arch>_gen.go for the instruction semantics and what was
// opened.

// Maximum computes dst[i] = max(a[i], b[i]) for i < len(a), the Go builtin max:
// NaN-propagating (the second operand's NaN if both are), +0 above -0, np.maximum
// up to the choice of NaN and the order of a (-0, +0) pair. It panics if len(b) <
// len(a) or len(dst) < len(a).
func Maximum(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Maximum: len(b) or len(dst) < len(a)")
	}
	vmaximum(dst[:len(a)], a, b[:len(a)])
}

// Minimum computes dst[i] = min(a[i], b[i]) for i < len(a), the Go builtin min:
// NaN-propagating (the second operand's NaN if both are), -0 below +0, np.minimum
// up to the choice of NaN and the order of a (-0, +0) pair. It panics if len(b) <
// len(a) or len(dst) < len(a).
func Minimum(dst, a, b []float64) {
	if len(b) < len(a) || len(dst) < len(a) {
		panic("floats: Minimum: len(b) or len(dst) < len(a)")
	}
	vminimum(dst[:len(a)], a, b[:len(a)])
}

// ---- portable reference kernels (the kernel on every architecture without one,
// and for the last elements of every call). The callers have sliced dst and b to
// len(a). They spell the rule out instead of calling the builtin, whose NaN
// choice is the target's.

func maxElem(x, y float64) float64 {
	if y != y || y > x {
		return y
	}
	if x != x || x > y || x != 0 {
		return x
	}
	// x and y are both ±0: -0 only if both are -0.
	return math.Float64frombits(math.Float64bits(x) & math.Float64bits(y))
}

func minElem(x, y float64) float64 {
	if y != y || y < x {
		return y
	}
	if x != x || x < y || x != 0 {
		return x
	}
	// x and y are both ±0: -0 if either is.
	return math.Float64frombits(math.Float64bits(x) | math.Float64bits(y))
}

func maximumRef(dst, a, b []float64) {
	dst, b = dst[:len(a)], b[:len(a)]
	for i, x := range a {
		dst[i] = maxElem(x, b[i])
	}
}

func minimumRef(dst, a, b []float64) {
	dst, b = dst[:len(a)], b[:len(a)]
	for i, x := range a {
		dst[i] = minElem(x, b[i])
	}
}
