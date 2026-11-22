package floats

import (
	"math"
	"unsafe"
)

// # Three-link all-pass decorrelator rows (Parametric Stereo)
//
// AllPass3 is the all-pass half of the decorrelator of MPEG-4 Parametric Stereo
// (ISO/IEC 14496-3, the HE-AAC v2 decoders): for each row k of a slot of complex
// samples, a fractional rotation Phi of the pre-delayed input followed by three
// Schroeder links in series, every link having its own delay line (the state of the
// link, a ring of past values) and its own rotation Q and real coefficient C:
//
//	x      = Phi[k] * in[k]
//	for l = 0, 1, 2:
//	    p  = Q_l[k] * old_l[k]          (the delayed state, rotated)
//	    c  = x + C_l[k]*p               (the new state of the link)
//	    cur_l[k] = c
//	    x  = p - C_l[k]*c               (the output of the link)
//	out[k] = x
//
// old_l is the slot the link delays back to, cur_l the slot it writes now: the caller
// resolves the ring pointers, the kernel only reads old_l and writes cur_l. Rows are
// independent, so the kernel runs two rows per AVX2 register (one per NEON register).
//
// A complex product here is the pair of operations
//
//	re = FMA(dr, ar, -(di*ai))      im = FMA(di, ar, dr*ai)
//
// for the coefficient a and the data d (the product di*ai or dr*ai rounded, the other
// fused into the sum), and c and x are each one FMA per component. The kernels
// (VPERMILPD, VMULPD, VFMADDSUB231PD on AVX2+FMA; VEXT, VFMUL, VFMLA on NEON) and the
// portable math.FMA loop do the same operations in the same order, so the result is
// bit-identical on every architecture and whatever CPU features are found at run time.
// It is not the unfused complex128 arithmetic of a Go loop on amd64, which rounds each
// product: a caller replacing such a loop may see last-bit differences.
//
// Error bound. With u = 2^-53 and exact arithmetic on the same float64 inputs, the
// complex error (the modulus of the difference) of every p, c and x is bounded by the
// recurrence below, in which |.| is the modulus of the exact value:
//
//	e(x_{-1}) <= 3u |Phi| |in|
//	e(p_l)    <= 3u |Q_l| |old_l|
//	e(c_l)    <= e(x_{l-1}) + |C_l| e(p_l) + u |c_l|
//	e(x_l)    <= e(p_l) + |C_l| e(c_l) + u |x_l|
//
// (a component of a complex product carries two roundings, each at most u times an
// operand magnitude; a fused sum carries one, at most u times its result). It is a few
// units of u times the largest magnitude in the chain for |C| <= 1. The tests assert it
// for out and for every cur_l against a 200-bit computation (TestAllPass3Bound,
// FuzzAllPass3).
//
// All sixteen operands are checked: in, old_l and cur_l must hold at least len(out)
// elements, as must each of the seven tables of t; nothing past len(out) is read or
// written. The written slices (out and cur_l) must not overlap one another or any
// operand that is read, not even exactly: checked, panics with a message saying so.
// len(out) = 0 does nothing.

// AllPassRows holds the constants of the rows of a layout: Phi the rotation of the
// pre-delay, Q_l the rotation of link l, C_l its real coefficient.
type AllPassRows struct {
	Phi, Q0, Q1, Q2 []complex128
	C0, C1, C2      []float64
}

// AllPass3 computes the decorrelator rows described above for k < len(out). It panics if
// an operand is shorter than len(out) or a written slice overlaps another operand.
func AllPass3(out, in []complex128, t *AllPassRows, old, cur *[3][]complex128) {
	n := len(out)
	if n == 0 {
		return
	}
	if allPassShort(n, in, t, old, cur) {
		panic("floats: AllPass3: an operand is shorter than len(out)")
	}
	if allPassOverlap(n, out, in, t, old, cur) {
		panic("floats: AllPass3: a written slice overlaps another operand")
	}
	allPass3(n, out, in, t, old, cur)
}

// allPassShort reports whether any operand other than out holds fewer than n elements.
func allPassShort(n int, in []complex128, t *AllPassRows, old, cur *[3][]complex128) bool {
	return len(in) < n ||
		len(t.Phi) < n || len(t.Q0) < n || len(t.Q1) < n || len(t.Q2) < n ||
		len(t.C0) < n || len(t.C1) < n || len(t.C2) < n ||
		len(old[0]) < n || len(old[1]) < n || len(old[2]) < n ||
		len(cur[0]) < n || len(cur[1]) < n || len(cur[2]) < n
}

// span is the byte range [lo, hi) of the first n elements of an operand.
// The addresses are widened to int64, so a difference of two of them is exact on every
// word size.
type span struct{ lo, hi int64 }

// meets is a.lo < b.hi && b.lo < a.hi as the sign of one value: both differences are
// negative exactly when the ranges share a byte, and the AND of two int64 is negative
// exactly when both are. It has no branch, so a caller can OR many of them and test the
// sign once. (The operands are non-empty: allPassShort and n > 0 hold before it runs.)
func (a span) meets(b span) int64 { return (a.lo - b.hi) & (b.lo - a.hi) }

func cspan(s []complex128, n int) span {
	lo := int64(uint64(uintptr(unsafe.Pointer(unsafe.SliceData(s)))))
	return span{lo, lo + int64(n)*int64(unsafe.Sizeof(s[0]))}
}

func fspan(s []float64, n int) span {
	lo := int64(uint64(uintptr(unsafe.Pointer(unsafe.SliceData(s)))))
	return span{lo, lo + int64(n)*int64(unsafe.Sizeof(s[0]))}
}

// allPassOverlap reports whether a written operand (out, cur_l) meets another written
// one or any operand read.
func allPassOverlap(n int, out, in []complex128, t *AllPassRows, old, cur *[3][]complex128) bool {
	w := [4]span{cspan(out, n), cspan(cur[0], n), cspan(cur[1], n), cspan(cur[2], n)}
	r := [11]span{cspan(in, n), cspan(old[0], n), cspan(old[1], n), cspan(old[2], n),
		cspan(t.Phi, n), cspan(t.Q0, n), cspan(t.Q1, n), cspan(t.Q2, n),
		fspan(t.C0, n), fspan(t.C1, n), fspan(t.C2, n)}
	var hit int64
	// indexed, not ranged over the arrays: a range copies them, and the copy reads 16
	// bytes across two 8-byte stores just made, a store-forwarding stall (measured in
	// go-avkit/decode#77, a third of this function's time)
	for i := range w {
		for j := i + 1; j < len(w); j++ {
			hit |= w[i].meets(w[j])
		}
		for j := range r {
			hit |= w[i].meets(r[j])
		}
	}
	return hit < 0
}

// cmulCoef is the product of the coefficient a and the data d as the kernels compute it:
// the products di*ai and dr*ai rounded, the others fused into the sum.
func cmulCoef(a, d complex128) complex128 {
	ar, ai, dr, di := real(a), imag(a), real(d), imag(d)
	return complex(math.FMA(dr, ar, -(di*ai)), math.FMA(di, ar, dr*ai))
}

// allPassLink is one link: the delayed state old rotated by q gives p, the new state is
// c = x + cl*p (written to *cur) and the output of the link is p - cl*c.
func allPassLink(x, q, old complex128, cl float64, cur *complex128) complex128 {
	p := cmulCoef(q, old)
	c := complex(math.FMA(cl, real(p), real(x)), math.FMA(cl, imag(p), imag(x)))
	*cur = c
	return complex(math.FMA(-cl, real(c), real(p)), math.FMA(-cl, imag(c), imag(p)))
}

// allPass3Ref is the portable loop and the reference of the kernels.
func allPass3Ref(n int, out, in []complex128, t *AllPassRows, old, cur *[3][]complex128) {
	for k := 0; k < n; k++ {
		x := cmulCoef(t.Phi[k], in[k])
		x = allPassLink(x, t.Q0[k], old[0][k], t.C0[k], &cur[0][k])
		x = allPassLink(x, t.Q1[k], old[1][k], t.C1[k], &cur[1][k])
		x = allPassLink(x, t.Q2[k], old[2][k], t.C2[k], &cur[2][k])
		out[k] = x
	}
}
